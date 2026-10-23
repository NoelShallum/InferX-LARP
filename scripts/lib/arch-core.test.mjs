import { describe, expect, it } from 'vitest';
import { archFromConfig, kvBytesPerSeq, layerKinds, towerParams } from './arch-core.mjs';

const llama = { model_type: 'llama', num_hidden_layers: 80, hidden_size: 8192, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 128, intermediate_size: 28672, vocab_size: 128256 };

describe('architecture from a config', () => {
  it('counts a plain GQA model at 2 × layers × KV heads × head dim bytes a token', () => {
    const a = archFromConfig(llama, { totalParams: 70.55e9 });
    expect(a.fullTokBytes).toBe(80 * 2 * 8 * 128);
    expect(a.stateBytes).toBe(0);
    expect(a.moe).toBeNull();
    expect(a.family).toBe('gqa');
  });
  it('subtracts the embedding lookup table from the active parameters of an untied dense model', () => {
    const a = archFromConfig(llama, { totalParams: 70.55e9 });
    expect(a.params.active).toBeCloseTo(70.55e9 - 128256 * 8192, -6);
  });
  it('keeps a tied embedding in the active parameters, since it doubles as the output head', () => {
    const a = archFromConfig({ ...llama, tie_word_embeddings: true }, { totalParams: 70.55e9 });
    expect(a.params.active).toBe(70.55e9);
  });
  it('gives only the full-attention layers of a hybrid model a KV cache, and the rest a recurrent state', () => {
    const layer_types = Array.from({ length: 8 }, (_, i) => (i % 4 === 3 ? 'full_attention' : 'linear_attention'));
    const a = archFromConfig({
      text_config: { num_hidden_layers: 8, hidden_size: 1024, num_attention_heads: 8, num_key_value_heads: 2, head_dim: 128, layer_types, linear_num_value_heads: 16, linear_num_key_heads: 4, linear_key_head_dim: 64, linear_value_head_dim: 64, linear_conv_kernel_dim: 4, vocab_size: 1000 },
    });
    expect(a.layerMix).toMatchObject({ full: 2, linear: 6 });
    expect(a.fullTokBytes).toBe(2 * 2 * 2 * 128);
    expect(a.stateBytes).toBeGreaterThan(6 * 16 * 64 * 64 * 4);
    expect(a.family).toBe('hybrid-linear');
  });
  it('caps sliding-window layers at the window', () => {
    const a = archFromConfig({ num_hidden_layers: 4, hidden_size: 512, num_attention_heads: 8, num_key_value_heads: 2, head_dim: 64, sliding_window: 128, layer_types: ['sliding_attention', 'full_attention', 'sliding_attention', 'full_attention'], vocab_size: 100 });
    const at = (n) => kvBytesPerSeq(a, n);
    expect(at(1000) - at(500)).toBe(500 * a.fullTokBytes);
    expect(a.window).toBe(128);
  });
  it('finds the routed experts and the share of the file they take', () => {
    const a = archFromConfig(
      { num_hidden_layers: 10, hidden_size: 1000, num_attention_heads: 8, num_key_value_heads: 8, head_dim: 125, num_local_experts: 16, num_experts_per_tok: 2, intermediate_size: 500, vocab_size: 10, tie_word_embeddings: true },
      { totalParams: 250_000_000 },
    );
    expect(a.moe.routedParams).toBe(10 * 16 * 3 * 1000 * 500);
    expect(a.params.routedShare).toBeCloseTo(240e6 / 250e6, 6);
    expect(a.params.active).toBeCloseTo(250e6 - 240e6 * (1 - 2 / 16), 0);
  });
  it('treats a latent mixture of experts as narrower than the hidden size, with two matrices for squared-ReLU experts', () => {
    const a = archFromConfig({ layers_block_type: ['mamba', 'moe', 'attention', 'moe'], hidden_size: 4096, num_attention_heads: 32, num_key_value_heads: 2, head_dim: 128, n_routed_experts: 8, num_experts_per_tok: 2, moe_intermediate_size: 100, moe_latent_size: 64, mlp_hidden_act: 'relu2', mamba_num_heads: 4, mamba_head_dim: 8, ssm_state_size: 16, n_groups: 1, vocab_size: 10 });
    expect(a.moe.routedParams).toBe(2 * 8 * 2 * 64 * 100);
    expect(a.layerMix).toMatchObject({ mamba: 1, full: 1, other: 2 });
    expect(a.stateBytes).toBeGreaterThan(4 * 8 * 16 * 4 - 1);
  });
  it('uses the latent width for multi-head latent attention', () => {
    const a = archFromConfig({ num_hidden_layers: 4, hidden_size: 256, num_attention_heads: 4, num_key_value_heads: 4, head_dim: 64, kv_lora_rank: 32, qk_rope_head_dim: 8, vocab_size: 10 });
    expect(a.fullTokBytes).toBe(4 * (32 + 8));
    expect(a.mla).toEqual({ latent: 32, rope: 8 });
  });
  it('reads the language model out of a nested config', () => {
    expect(layerKinds({ num_hidden_layers: 3 })).toEqual(['full', 'full', 'full']);
    const a = archFromConfig({ llm_config: llama });
    expect(a.layers).toBe(80);
  });
  it('leaves vision and audio towers out of the text parameters when the count includes them', () => {
    const vision = { depth: 10, hidden_size: 100, intermediate_size: 400 };
    expect(towerParams(vision)).toBe(10 * (4 * 100 * 100 + 2 * 100 * 400));
    const a = archFromConfig({ ...llama, vision_config: vision }, { totalParams: 70.55e9 });
    expect(a.params.text).toBe(70.55e9 - towerParams(vision));
    const b = archFromConfig({ ...llama, vision_config: vision }, { totalParams: 70.55e9, towersInTotal: false });
    expect(b.params.text).toBe(70.55e9);
  });
});
