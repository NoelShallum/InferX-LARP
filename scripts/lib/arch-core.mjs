/**
 * Architecture facts from a Hugging Face config.json: how many bytes a context token and a running
 * sequence cost in memory, and how much of the weight file a decode step has to read.
 *
 * Everything here is arithmetic on the config. The cache is counted at one byte per element (FP8 KV
 * cache), which is how the report derived its own KV figures. The recurrent state of linear-attention
 * and Mamba layers is counted in fp32, as the configs ask for.
 */

export const MODEL_REPOS = {
  qwen38_27b: { repo: 'Qwen/Qwen3.8-27B' },
  qwen36_35b: { repo: 'Qwen/Qwen3.6-35B-A3B' },
  gemma4_26b: { repo: 'google/gemma-4-26B-A4B-it' },
  gemma4_31b: { repo: 'google/gemma-4-31B-it' },
  gptoss_120b: { repo: 'openai/gpt-oss-120b' },
  gptoss_20b: { repo: 'openai/gpt-oss-20b' },
  // meta-llama's own repository is gated; this mirror carries the same config.
  llama33_70b: { repo: 'unsloth/Llama-3.3-70B-Instruct', note: 'Config read from an ungated mirror of meta-llama/Llama-3.3-70B-Instruct' },
  mistral_nemo: { repo: 'mistralai/Mistral-Nemo-Instruct-2407' },
  flashnext: { repo: 'Qwen/Qwen3.8-Flash-Next' },
  nemotron35: {
    repo: 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16',
    paramsRepo: 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16',
    note: 'Config from the 3.5 Super VL repository (its language backbone). Parameter count from the text-only Nemotron 3 Super, which has the same 88-layer backbone, so the vision and audio towers are not counted.',
  },
  mistral_small4: { repo: 'mistralai/Mistral-Small-4-119B-2603' },
  devstral2: { repo: 'mistralai/Devstral-2-123B-Instruct-2512' },
  qwen35_122b: { repo: 'Qwen/Qwen3.5-122B-A10B' },
  // Not a catalog model: it only calibrates the roofline, because InferenceX measured it on four PRO 6000s.
  qwen35_397b: { repo: 'Qwen/Qwen3.5-397B-A17B', note: 'Calibration only: InferenceX measured this model on four RTX PRO 6000s, the one measured multi-GPU PCIe data for that card' },
};

const num = (x) => (typeof x === 'number' && Number.isFinite(x) ? x : null);

/** The language-model section of a config, whichever way the repository nests it. */
export function textConfig(config) {
  return config.text_config ?? config.llm_config ?? config;
}

/** One entry per layer: 'full', 'sliding', 'linear' (gated delta net), 'mamba', or 'moe-only' (no attention). */
export function layerKinds(t) {
  if (Array.isArray(t.layers_block_type)) {
    return t.layers_block_type.map((k) => (k === 'attention' ? 'full' : k === 'mamba' ? 'mamba' : 'moe-only'));
  }
  const n = num(t.num_hidden_layers) ?? 0;
  if (Array.isArray(t.layer_types)) {
    return t.layer_types.map((k) => (k === 'full_attention' ? 'full' : k === 'sliding_attention' ? 'sliding' : k === 'linear_attention' ? 'linear' : 'full'));
  }
  return Array.from({ length: n }, () => 'full');
}

/** Rough parameter count of a vision or audio tower: layers × (attention 4h² + two MLP matrices). */
export function towerParams(c) {
  if (!c || typeof c !== 'object') return 0;
  const layers = num(c.depth) ?? num(c.num_hidden_layers) ?? num(c.num_layers);
  const h = num(c.hidden_size);
  const inter = num(c.intermediate_size);
  if (!layers || !h || !inter) return 0;
  return layers * (4 * h * h + 2 * h * inter);
}

/**
 * `towersInTotal`: the parameter count includes the vision and audio towers, which a text-only server does not read.
 */
export function archFromConfig(config, { totalParams = null, tiedFallback = false, towersInTotal = true } = {}) {
  const t = textConfig(config);
  const kinds = layerKinds(t);
  const count = (k) => kinds.filter((x) => x === k).length;
  const nFull = count('full');
  const nSliding = count('sliding');
  const nLinear = count('linear');
  const nMamba = count('mamba');
  const hidden = t.hidden_size;
  const vocab = t.vocab_size;
  const tied = Boolean(config.tie_word_embeddings ?? t.tie_word_embeddings ?? tiedFallback);
  const heads = t.num_attention_heads;
  const kvHeads = t.num_key_value_heads ?? heads;
  const headDim = t.head_dim ?? (hidden && heads ? hidden / heads : 0);
  const kEqV = Boolean(t.attention_k_eq_v);
  const mla = num(t.kv_lora_rank) ? { latent: t.kv_lora_rank, rope: t.qk_rope_head_dim ?? 0 } : null;

  // KV bytes per context token, summed over layers, at one byte per element.
  let fullTokBytes;
  if (mla) {
    fullTokBytes = nFull * (mla.latent + mla.rope);
  } else if (num(t.global_head_dim) && num(t.num_global_key_value_heads)) {
    fullTokBytes = nFull * (kEqV ? 1 : 2) * t.num_global_key_value_heads * t.global_head_dim;
  } else {
    fullTokBytes = nFull * 2 * kvHeads * headDim;
  }
  const slidingTokBytes = nSliding * 2 * kvHeads * headDim;
  const window = nSliding > 0 ? num(t.sliding_window) : null;

  // Recurrent state, per sequence.
  let stateBytes = 0;
  const recurrent = [];
  if (nLinear > 0) {
    const nv = t.linear_num_value_heads;
    const nk = t.linear_num_key_heads;
    const dk = t.linear_key_head_dim;
    const dv = t.linear_value_head_dim;
    const ssm = nv * dk * dv * 4;
    const conv = (2 * nk * dk + nv * dv) * Math.max(0, (t.linear_conv_kernel_dim ?? 4) - 1) * 2;
    stateBytes += nLinear * (ssm + conv);
    recurrent.push(`${nLinear} gated-delta-net layers × ${((ssm + conv) / 1e6).toFixed(2)} MB`);
  }
  if (nMamba > 0) {
    const dInner = t.mamba_num_heads * t.mamba_head_dim;
    const ssm = t.mamba_num_heads * t.mamba_head_dim * t.ssm_state_size * (t.mamba_ssm_cache_dtype === 'float16' || t.mamba_ssm_cache_dtype === 'bfloat16' ? 2 : 4);
    const conv = (dInner + 2 * t.n_groups * t.ssm_state_size) * Math.max(0, (t.conv_kernel ?? 4) - 1) * 2;
    stateBytes += nMamba * (ssm + conv);
    recurrent.push(`${nMamba} Mamba layers × ${((ssm + conv) / 1e6).toFixed(2)} MB`);
  }

  // Sparse attention: a few models only read a fixed budget of cached tokens per step.
  const readCapTokens = num(t.indexer_budget);

  // Mixture of experts.
  const experts = num(t.num_experts) ?? num(t.num_local_experts) ?? num(t.n_routed_experts);
  const topK = num(t.num_experts_per_tok) ?? num(t.top_k_experts);
  let moe = null;
  if (experts && topK) {
    const moeLayers = Array.isArray(t.layers_block_type) ? t.layers_block_type.filter((k) => k === 'moe').length : t.num_hidden_layers - (num(t.first_k_dense_replace) ?? 0);
    const inter = t.moe_intermediate_size ?? t.intermediate_size;
    const matrices = t.mlp_hidden_act === 'relu2' ? 2 : 3;
    const dimIn = num(t.moe_latent_size) ?? hidden;
    moe = { experts, topK, moeLayers, expertInter: inter, matrices, dimIn, routedParams: moeLayers * experts * matrices * dimIn * inter };
  }

  const embed = vocab * hidden;
  const mtpLayers = num(t.mtp_num_hidden_layers) ?? num(t.num_nextn_predict_layers) ?? 0;
  let params = null;
  if (totalParams) {
    // Parts of the file a decode step never reads: vision and audio towers, the speculative-decoding head, n-gram lookup tables.
    const towers = towersInTotal ? towerParams(config.vision_config) + towerParams(config.audio_config) : 0;
    const mtpExperts = moe && mtpLayers ? mtpLayers * (moe.routedParams / moe.moeLayers) : 0;
    const ngram = num(t.ngram_vocab_size_base) ? t.ngram_vocab_size_base * hidden : 0;
    const text = totalParams - towers - mtpExperts - ngram;
    // The embedding table is a lookup, so only its row is read; a tied table doubles as the output head and is read in full.
    const lookupOnly = tied ? 0 : embed;
    const routedInactive = moe ? moe.routedParams * (1 - moe.topK / moe.experts) : 0;
    params = {
      total: totalParams,
      text,
      active: text - lookupOnly - routedInactive,
      embed,
      embedTied: tied,
      routedShare: moe ? Math.min(1, moe.routedParams / text) : 0,
      excluded: { towers, mtpExperts, ngram },
    };
  }

  const family = nMamba > 0 ? 'hybrid-mamba-moe' : nLinear > 0 ? (moe ? 'hybrid-linear-moe' : 'hybrid-linear') : mla ? 'mla-moe' : nSliding > 0 ? (moe ? 'sliding-moe' : 'sliding') : moe ? 'gqa-moe' : 'gqa';
  return {
    family,
    layers: kinds.length,
    layerMix: { full: nFull, sliding: nSliding, linear: nLinear, mamba: nMamba, other: kinds.length - nFull - nSliding - nLinear - nMamba },
    hidden,
    vocab,
    heads,
    kvHeads,
    headDim,
    kEqV,
    mla,
    window,
    fullTokBytes,
    slidingTokBytes,
    stateBytes,
    stateDetail: recurrent.join(' + '),
    readCapTokens,
    moe,
    params,
    mtpLayers,
    maxPositions: num(t.max_position_embeddings),
  };
}

/** Bytes a sequence of `ctx` tokens holds in the cache. */
export function kvBytesPerSeq(a, ctx, bytesPerElt = 1) {
  const sliding = a.window ? Math.min(ctx, a.window) : ctx;
  return (a.fullTokBytes * ctx + a.slidingTokBytes * sliding) * bytesPerElt + a.stateBytes;
}
