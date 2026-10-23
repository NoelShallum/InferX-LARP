/**
 * Architecture facts from a Hugging Face config.json: how many bytes a context token and a running
 * sequence cost in memory, and how much of the weight file a decode step has to read.
 *
 * Everything here is arithmetic on the config. The cache is counted at one byte per element (FP8 KV
 * cache), which is how the report derived its own KV figures. The recurrent state of linear-attention
 * and Mamba layers is counted in fp32, as the configs ask for.
 *
 * A few newer models describe their attention in ways a config cannot spell out. For those, `card` carries the
 * figures their model cards state, and the result says which source each figure came from.
 */

const CARD = (repo) => `https://huggingface.co/${repo}`;

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

  // The highest-traffic models on OpenRouter. Their model cards state parameter counts the configs cannot give.
  dsv41flash: {
    repo: 'deepseek-ai/DeepSeek-V4.1-Flash',
    note: 'DeepSeek-V4.1-Flash is a causal encoder-decoder with Compressed Sparse Attention 2 and a 196B-parameter Engram lookup table. Parameter counts, the Engram size and the KV cache per token are the model card\'s figures; the sliding-window part of the cache is an estimate.',
    card: {
      source: CARD('deepseek-ai/DeepSeek-V4.1-Flash'),
      totalParams: 552e9,
      activeParams: 16e9,
      activePrefillParams: 8e9,
      lookupParams: 196e9,
      lookupGB: 196,
      // "reduce the global KV cache footprint to 890 bytes per token", with FP4 main KV
      fullTokBytes: 890,
      kvNote: 'Model card: 890 bytes of global KV per token with an FP4 main cache. The sliding-window state (128 tokens, one 512-wide KV head per layer) is an estimate.',
      slidingTokBytes: 40 * 512,
      window: 128,
    },
  },
  glm53: { repo: 'zai-org/GLM-5.3', note: 'Multi-head latent attention with a DeepSeek-style sparse indexer. The indexer cache (one 128-wide key per token per layer) is an assumption.' },
  glm53flash: {
    repo: 'zai-org/GLM-5.3-Flash',
    note: 'Hybrid of 34 Kimi-delta linear-attention layers and 11 sparse latent-attention layers. The indexer cache (one 128-wide key per token per layer) is an assumption.',
    card: { source: CARD('zai-org/GLM-5.3-Flash'), totalParams: 320e9, activeParams: 18e9 },
  },
  minimaxm3: {
    repo: 'MiniMaxAI/MiniMax-M3',
    note: 'Grouped-query attention with MiniMax Sparse Attention. The index cache (4 heads of 128 per token on each sparse layer) and the block-sparse read budget come from the config; whether the index keeps values is an assumption.',
    card: { source: CARD('MiniMaxAI/MiniMax-M3'), totalParams: 428e9, activeParams: 23e9 },
  },
  mimo26flash: {
    repo: 'XiaomiMiMo/MiMo-V2.6-Flash-RL',
    note: '9 global-attention layers and 39 sliding-window layers of 128 tokens.',
    card: { source: CARD('XiaomiMiMo/MiMo-V2.6-Flash-RL'), totalParams: 309e9, activeParams: 15e9 },
  },
};

const num = (x) => (typeof x === 'number' && Number.isFinite(x) ? x : null);

/** The language-model section of a config, whichever way the repository nests it. */
export function textConfig(config) {
  return config.text_config ?? config.llm_config ?? config;
}

/** One entry per layer: 'full', 'sliding', 'linear' (gated delta net, Kimi delta), 'mamba', or 'moe-only' (no attention). */
export function layerKinds(t) {
  if (Array.isArray(t.layers_block_type)) {
    return t.layers_block_type.map((k) => (k === 'attention' ? 'full' : k === 'mamba' ? 'mamba' : 'moe-only'));
  }
  const n = num(t.num_hidden_layers) ?? 0;
  if (Array.isArray(t.layer_types)) {
    return t.layer_types.map((k) => (k === 'full_attention' ? 'full' : k === 'sliding_attention' ? 'sliding' : k === 'linear_attention' ? 'linear' : 'full'));
  }
  // A 0/1 pattern with a sliding window: 1 marks the sliding-window layers.
  if (Array.isArray(t.hybrid_layer_pattern) && num(t.sliding_window)) {
    return t.hybrid_layer_pattern.map((k) => (k === 1 ? 'sliding' : 'full'));
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

/** Number of layers whose feed-forward is a mixture of experts. */
function moeLayerCount(t) {
  if (Array.isArray(t.layers_block_type)) return t.layers_block_type.filter((k) => k === 'moe').length;
  if (Array.isArray(t.moe_layer_freq)) return t.moe_layer_freq.filter((x) => x === 1).length;
  return t.num_hidden_layers - (num(t.first_k_dense_replace) ?? 0);
}

/**
 * `towersInTotal`: the parameter count includes the vision and audio towers, which a text-only server does not read.
 * `card`: figures from the model card, which win over what the config implies.
 */
export function archFromConfig(config, { totalParams = null, tiedFallback = false, towersInTotal = true, card = null } = {}) {
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
  const headDim = t.head_dim || (hidden && heads ? hidden / heads : 0);
  const vDim = num(t.v_head_dim) ?? headDim;
  const kEqV = Boolean(t.attention_k_eq_v);
  const mla = num(t.kv_lora_rank) ? { latent: t.kv_lora_rank, rope: t.qk_rope_head_dim ?? 0 } : null;

  // KV bytes per context token, summed over layers, at one byte per element.
  let fullTokBytes;
  let perFullLayer;
  if (mla) {
    perFullLayer = mla.latent + mla.rope;
  } else if (num(t.global_head_dim) && num(t.num_global_key_value_heads)) {
    perFullLayer = (kEqV ? 1 : 2) * t.num_global_key_value_heads * t.global_head_dim;
  } else {
    perFullLayer = kvHeads * (headDim + vDim);
  }
  fullTokBytes = nFull * perFullLayer;
  const swaHeads = num(t.swa_num_key_value_heads) ?? kvHeads;
  const swaK = num(t.swa_head_dim) ?? headDim;
  const swaV = num(t.swa_v_head_dim) ?? vDim;
  const slidingTokBytes = nSliding * swaHeads * (swaK + swaV);
  const window = nSliding > 0 ? num(t.sliding_window) : null;

  // Sparse attention reads only part of the cache, and keeps an index of every token it can pick from.
  let readCapTokens = num(t.indexer_budget);
  let cappedTokBytes = readCapTokens ? fullTokBytes : 0;
  let scanTokBytes = 0;
  const notes = [];
  if (mla && num(t.index_topk) && num(t.index_head_dim)) {
    // DeepSeek-style sparse indexer: a 128-wide key per token per layer, and attention over the top-k tokens.
    readCapTokens = t.index_topk;
    cappedTokBytes = fullTokBytes;
    scanTokBytes = nFull * t.index_head_dim;
    notes.push(`sparse indexer: ${nFull} layers × ${t.index_head_dim} bytes of index key per token (assumed one head), attention over the top ${t.index_topk} tokens`);
  }
  const msa = t.sparse_attention_config;
  if (msa && msa.use_sparse_attention && Array.isArray(msa.sparse_attention_freq)) {
    const sparseLayers = msa.sparse_attention_freq.filter((x) => x === 1).length;
    const blocks = (msa.sparse_topk_blocks ?? 0) + (msa.sparse_local_block ?? 0) + (msa.sparse_init_block ?? 0);
    readCapTokens = blocks * (msa.sparse_block_size ?? 128);
    cappedTokBytes = sparseLayers * perFullLayer;
    scanTokBytes = sparseLayers * (msa.sparse_num_index_heads ?? 1) * (msa.sparse_index_dim ?? 128);
    notes.push(`sparse attention on ${sparseLayers} of ${nFull} layers: ${blocks} blocks of ${msa.sparse_block_size ?? 128} tokens read, ${msa.sparse_num_index_heads} index heads of ${msa.sparse_index_dim} per token (keys only, assumed)`);
  }

  // Recurrent state, per sequence.
  let stateBytes = 0;
  const recurrent = [];
  if (nLinear > 0 && t.linear_attn_config) {
    const la = t.linear_attn_config;
    const ssm = la.num_heads * la.head_dim * la.head_dim * 4;
    const conv = 3 * la.num_heads * la.head_dim * Math.max(0, (la.short_conv_kernel_size ?? 4) - 1) * 2;
    stateBytes += nLinear * (ssm + conv);
    recurrent.push(`${nLinear} Kimi-delta layers × ${((ssm + conv) / 1e6).toFixed(2)} MB`);
  } else if (nLinear > 0) {
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

  // Mixture of experts.
  const experts = num(t.num_experts) ?? num(t.num_local_experts) ?? num(t.n_routed_experts);
  const topK = num(t.num_experts_per_tok) ?? num(t.top_k_experts);
  let moe = null;
  if (experts && topK) {
    const moeLayers = moeLayerCount(t);
    const inter = t.moe_intermediate_size ?? t.intermediate_size;
    const matrices = t.mlp_hidden_act === 'relu2' ? 2 : 3;
    const dimIn = num(t.moe_latent_size) ?? hidden;
    moe = { experts, topK, moeLayers, expertInter: inter, matrices, dimIn, routedParams: moeLayers * experts * matrices * dimIn * inter };
  }

  const embed = vocab * hidden;
  const mtpLayers = num(t.mtp_num_hidden_layers) ?? num(t.num_nextn_predict_layers) ?? 0;
  let params = null;
  const ngramParams = num(t.ngram_vocab_size_base) ? t.ngram_vocab_size_base * hidden : 0;
  if (card?.totalParams || totalParams) {
    // Parts of the file a decode step never reads: vision and audio towers, the speculative-decoding head, n-gram lookup tables.
    const towers = towersInTotal && !card?.totalParams ? towerParams(config.vision_config) + towerParams(config.audio_config) : 0;
    const mtpExperts = moe && mtpLayers && !card?.totalParams ? mtpLayers * (moe.routedParams / moe.moeLayers) : 0;
    const text = card?.totalParams ?? totalParams - towers - mtpExperts - ngramParams;
    // The embedding table is a lookup, so only its row is read; a tied table doubles as the output head and is read in full.
    const lookupOnly = tied ? 0 : embed;
    const routedInactive = moe ? moe.routedParams * (1 - moe.topK / moe.experts) : 0;
    params = {
      total: totalParams ?? card.totalParams,
      text,
      active: card?.activeParams ?? text - lookupOnly - routedInactive,
      activePrefill: card?.activePrefillParams ?? null,
      embed,
      embedTied: tied,
      routedShare: moe ? Math.min(1, moe.routedParams / text) : 0,
      excluded: { towers, mtpExperts, ngram: ngramParams },
      source: card?.source ?? null,
    };
  }

  // Weights that are lookup tables and never read in full: the n-gram table (BF16) or the Engram table.
  const lookupGB = card?.lookupGB ?? (ngramParams ? (ngramParams * 2) / 1e9 : 0);

  let out = {
    fullTokBytes,
    cappedTokBytes,
    scanTokBytes,
    slidingTokBytes,
    window,
    readCapTokens,
  };
  let kvSource = 'config';
  let kvNote = notes.join('; ');
  if (card?.fullTokBytes !== undefined) {
    out = { fullTokBytes: card.fullTokBytes, cappedTokBytes: 0, scanTokBytes: 0, slidingTokBytes: card.slidingTokBytes ?? 0, window: card.window ?? null, readCapTokens: null };
    kvSource = 'model-card';
    kvNote = card.kvNote ?? '';
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
    ...out,
    stateBytes,
    stateDetail: recurrent.join(' + '),
    lookupGB,
    kvSource,
    kvNote,
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
