import { FX_INR_PER_USD, GPUS, HOURS_PER_MONTH, SUMMER_DERATE } from './gpus';
import { CACHE_PRICE_FRACTION, MARKET } from './market';
import { BYTES_PER_PARAM, GOLDEN_AL_QWEN38_FLASH_NEXT, MODELS } from './models';
import { liveRegister } from './live';
import * as SC from './scenarios';
import { ROOF_PROV } from '../engine/roofline';
import { GPU_MEM_UTIL, OVERHEAD_GB_PER_GPU, MIN_KV_TOKENS, ROOFLINE_MAX_CONC, REPORT_TEST_CTX, DRAFT_COST, MTP_DEFAULT_CSAT, VOLTA_KERNEL_PENALTY, MAX_EXTRAPOLATION } from '../engine/capacity';

/** Every value the dashboard reads from its data and constants, for the Data audit and the tests that check it. */
export const ALL = {
  gpus: GPUS,
  models: MODELS,
  market: MARKET,
  scenarios: SC.SCENARIOS,
  lium: SC.LIUM,
  program: SC.PROGRAM,
  routeCosts: SC.ROUTE_COSTS,
  workloads: SC.WORKLOADS,
  availability: SC.AVAILABILITY_PRESETS,
  uptimeTiers: SC.UPTIME_TIERS,
  live: liveRegister(),
  constants: {
    fx: FX_INR_PER_USD,
    hoursPerMonth: HOURS_PER_MONTH,
    cachePriceFraction: CACHE_PRICE_FRACTION,
    summerDerate: SUMMER_DERATE,
    priceElasticity: SC.ELASTICITY_PRICE,
    latencyElasticity: SC.ELASTICITY_LATENCY,
    zdrBonus: SC.ZDR_SHARE_BONUS,
    ramp: SC.RAMP,
    floorMonths: SC.FLOOR_MONTHS,
    batchDiscount: SC.BATCH_DISCOUNT,
    gpuMemUtil: GPU_MEM_UTIL,
    overheadGbPerGpu: OVERHEAD_GB_PER_GPU,
    minKvTokens: MIN_KV_TOKENS,
    rooflineMaxConc: ROOFLINE_MAX_CONC,
    reportTestCtx: REPORT_TEST_CTX,
    roofline: ROOF_PROV,
    draftCost: DRAFT_COST,
    mtpDefaultCsat: MTP_DEFAULT_CSAT,
    voltaPenalty: VOLTA_KERNEL_PENALTY,
    maxExtrapolation: MAX_EXTRAPOLATION,
    bytesPerParam: BYTES_PER_PARAM,
    goldenAl: GOLDEN_AL_QWEN38_FLASH_NEXT,
    buyHorizon: SC.BUY_HORIZON_MONTHS,
    rentRules: SC.RENT_RULES,
    listingOdds: SC.LISTING_ODDS,
    secondModelDelay: SC.SECOND_MODEL_DELAY,
    tax: SC.TAX,
  },
};
