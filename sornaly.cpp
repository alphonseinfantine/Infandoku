#include <bits/stdc++.h>
#include <random>
#include <chrono>

#pragma GCC optimize("O3,unroll-loops,fast-math")
#pragma GCC target("avx2,fma")

using namespace std;

constexpr int N_CHANNELS    = 79;
constexpr int HISTORY_WIN   = 4;   // [V47] Fenêtre courte = cache-friendly
constexpr int CACHE_TTL     = 8;
constexpr int MAX_BLOCK     = 12;
constexpr float DECAY       = 0.9f;
constexpr int TOTAL_TIME    = 1000 * MAX_BLOCK;
constexpr int SURV_INTERVAL_A = 20;
constexpr int SURV_INTERVAL_B = 15;
constexpr int SURV_INTERVAL_C = 10;
constexpr int BANDIT_MIN_OBS = 5;
constexpr float BANDIT_DECAY = 0.92f;
constexpr int WARM_HOPS     = 15;
constexpr int WEIBULL_MIN_SAMPLES = 5;
constexpr int BANDIT_DECAY_FLUSH = 32;  // [V47] 32 reste optimal ici
constexpr float IIR_ALPHA   = 0.75f;
constexpr float WEIBULL_EMA_ALPHA = 0.85f;
constexpr int META_WINDOW   = 20;

struct WeibullLookup {
    static constexpr int N_ENTRIES = 6;
    float cv_thresholds[N_ENTRIES];
    float k_values[N_ENTRIES];
    float gamma_values[N_ENTRIES];
    WeibullLookup() {
        cv_thresholds[0] = 0.35f;  k_values[0] = 3.0f;  gamma_values[0] = 0.8930f;
        cv_thresholds[1] = 0.55f;  k_values[1] = 2.0f;  gamma_values[1] = 0.8862f;
        cv_thresholds[2] = 0.85f;  k_values[2] = 1.5f;  gamma_values[2] = 0.9027f;
        cv_thresholds[3] = 1.25f;  k_values[3] = 1.0f;  gamma_values[3] = 1.0f;
        cv_thresholds[4] = 1.60f;  k_values[4] = 0.7f;  gamma_values[4] = 1.2981f;
        cv_thresholds[5] = 999.0f; k_values[5] = 0.5f;  gamma_values[5] = 2.0f;
    }
    inline void lookup(float cv, float& out_k, float& out_gamma) const {
        int idx = 0;
        while (cv > cv_thresholds[idx]) idx++;
        out_k = k_values[idx];
        out_gamma = gamma_values[idx];
    }
};
static const WeibullLookup WEIBULL_LUT;

struct SpectralEnvironment {
    vector<array<uint64_t, 2>> occupancy;
    SpectralEnvironment(int seed = 42) {
        mt19937 rng(seed);
        occupancy.resize(TOTAL_TIME);
        for (auto& slot : occupancy) slot = {0, 0};
        vector<int> wifi_channels = {1, 6, 11};
        for (int wch : wifi_channels) {
            int center = wch * 6;
            int s_ch = max(0, center - 11);
            int e_ch = min(N_CHANNELS, center + 11);
            int t = 0;
            while (t < TOTAL_TIME) {
                int burst = uniform_int_distribution<int>(20, 50)(rng);
                int gap   = uniform_int_distribution<int>(5, 15)(rng);
                for (int dt = 0; dt < burst && t + dt < TOTAL_TIME; ++dt)
                    for (int ch = s_ch; ch < e_ch; ++ch)
                        set_occupancy(ch, t + dt);
                t += burst + gap;
            }
        }
        for (int p = 0; p < 4; ++p) {
            int ch = uniform_int_distribution<int>(0, N_CHANNELS - 1)(rng);
            int t = 0;
            while (t < TOTAL_TIME) {
                int dwell = uniform_int_distribution<int>(5, 15)(rng);
                for (int dt = 0; dt < dwell && t + dt < TOTAL_TIME; ++dt)
                    set_occupancy(ch, t + dt);
                ch = clamp(ch + uniform_int_distribution<int>(-5, 5)(rng), 0, N_CHANNELS - 1);
                t += dwell;
            }
        }
        uniform_real_distribution<float> dist01(0.0f, 1.0f);
        for (int t = 0; t < TOTAL_TIME; ++t)
            for (int ch = 0; ch < N_CHANNELS; ++ch)
                if (dist01(rng) < 0.10f) set_occupancy(ch, t);
    }
    inline void set_occupancy(int ch, int t) {
        if (ch < 64) occupancy[t][0] |= (1ULL << ch);
        else         occupancy[t][1] |= (1ULL << (ch - 64));
    }
    inline bool is_collision(int ch, int t) const {
        if (t >= TOTAL_TIME) return false;
        if (ch < 64) return (occupancy[t][0] >> ch) & 1ULL;
        return (occupancy[t][1] >> (ch - 64)) & 1ULL;
    }
    void observe_single(int t, float* ring_buffer, int head_idx) const {
        uint64_t b0 = occupancy[t][0];
        uint64_t b1 = occupancy[t][1];
        int col = head_idx;
        for (int ch = 0; ch < 64; ++ch) {
            ring_buffer[ch * HISTORY_WIN + col] = (float)(b0 & 1);
            b0 >>= 1;
        }
        for (int ch = 64; ch < N_CHANNELS; ++ch) {
            ring_buffer[ch * HISTORY_WIN + col] = (float)(b1 & 1);
            b1 >>= 1;
        }
    }
    pair<uint64_t, uint64_t> get_current_free(int t) const {
        if (t >= TOTAL_TIME) return {~0ULL, ~0ULL};
        return {~occupancy[t][0], ~occupancy[t][1]};
    }
};

class SornalyV49 {
public:
    alignas(64) float ring_buffer[N_CHANNELS * HISTORY_WIN];
    alignas(64) float occ_cache[N_CHANNELS];
    alignas(64) float var_cache[N_CHANNELS];
    alignas(64) float score_buffer[N_CHANNELS];
    alignas(64) float weights[HISTORY_WIN];
    alignas(64) float buf_s_occ[N_CHANNELS];
    alignas(64) float buf_s_surv[N_CHANNELS];
    alignas(64) float buf_s_var[N_CHANNELS];
    uint8_t collision_ring[32];
    int     collision_ring_idx  = 0;
    int     collision_ring_size = 0;
    uint8_t channel_collision_hist[N_CHANNELS][16];
    uint8_t channel_hist_idx[N_CHANNELS];
    uint8_t channel_hist_size[N_CHANNELS];
    alignas(64) float bandit_alpha[3][N_CHANNELS];
    alignas(64) float bandit_beta[3][N_CHANNELS];
    alignas(64) int   bandit_obs_count[3][N_CHANNELS];
    int bandit_decay_pending[3] = {0, 0, 0};
    int excluded_ttl[N_CHANNELS];
    vector<int> hop_sequence;
    int current_block[MAX_BLOCK];
    int block_size = 0;
    int block_idx  = 0;
    uint16_t current_idle[N_CHANNELS];
    uint8_t  last_state[N_CHANNELS];
    float lambda_weibull[N_CHANNELS];
    float shape_k[N_CHANNELS];
    bool  weibull_valid[N_CHANNELS];
    float weibull_mean[N_CHANNELS];
    float weibull_var[N_CHANNELS];
    int   weibull_count[N_CHANNELS];
    float cache_surv_1[N_CHANNELS];
    bool  cache_surv_valid = false;
    float threshold       = 0.25f;
    int   base_cooldown   = 8;
    int   hops_done       = 0;
    int   collisions      = 0;
    int   backtracks      = 0;
    int   cache_t         = -9999;
    int   surv_t          = -1;
    int safe_channels[N_CHANNELS];
    int n_safe = 0;
    float score_weights[3][3];
    int   stat_A = 0, stat_B = 0, stat_C = 0, stat_fallback = 0;
    mt19937 rng;
    const SpectralEnvironment& env;
    int surv_skipped       = 0;
    int total_bandit_calls = 0;
    float ema_ratio = 0.5f;
    float ema_alpha = 0.1f;
    float performance_history[META_WINDOW];
    int   perf_hist_idx  = 0;
    int   perf_hist_size = 0;
    float threshold_momentum = 0.0f;
    char  last_regime_char = 'A';
    float last_regime_ratio = 0.5f;
    bool  regime_cache_first = true;
    int ring_head = 0;

    alignas(64) float pre_s_occ[N_CHANNELS];
    alignas(64) float pre_s_var[N_CHANNELS];
    alignas(64) float pre_s_surv_h1[N_CHANNELS];
    bool pre_scores_valid = false;

    SornalyV49(const SpectralEnvironment& e, int seed = 42) : env(e), rng(seed) {
        hop_sequence.reserve(10000);
        memset(ring_buffer,            0, sizeof(ring_buffer));
        memset(occ_cache,             0, sizeof(occ_cache));
        memset(var_cache,             0, sizeof(var_cache));
        memset(score_buffer,          0, sizeof(score_buffer));
        memset(buf_s_occ,            0, sizeof(buf_s_occ));
        memset(buf_s_surv,           0, sizeof(buf_s_surv));
        memset(buf_s_var,            0, sizeof(buf_s_var));
        memset(collision_ring,        0, sizeof(collision_ring));
        memset(channel_collision_hist,0, sizeof(channel_collision_hist));
        memset(channel_hist_idx,      0, sizeof(channel_hist_idx));
        memset(channel_hist_size,     0, sizeof(channel_hist_size));
        memset(bandit_alpha,          0, sizeof(bandit_alpha));
        memset(bandit_beta,           0, sizeof(bandit_beta));
        memset(bandit_obs_count,      0, sizeof(bandit_obs_count));
        memset(excluded_ttl,          0, sizeof(excluded_ttl));
        memset(safe_channels,         0, sizeof(safe_channels));
        memset(current_block,         0, sizeof(current_block));
        memset(current_idle,          0, sizeof(current_idle));
        memset(last_state,            1, sizeof(last_state));
        memset(lambda_weibull,        0, sizeof(lambda_weibull));
        memset(shape_k,               0, sizeof(shape_k));
        memset(weibull_valid,         0, sizeof(weibull_valid));
        memset(cache_surv_1,          0, sizeof(cache_surv_1));
        memset(performance_history,   0, sizeof(performance_history));
        memset(weibull_mean,          0, sizeof(weibull_mean));
        memset(weibull_var,           0, sizeof(weibull_var));
        memset(weibull_count,         0, sizeof(weibull_count));
        memset(pre_s_occ,            0, sizeof(pre_s_occ));
        memset(pre_s_var,            0, sizeof(pre_s_var));
        memset(pre_s_surv_h1,       0, sizeof(pre_s_surv_h1));

        for (int r = 0; r < 3; ++r)
            for (int ch = 0; ch < N_CHANNELS; ++ch) {
                bandit_alpha[r][ch] = 1.0f;
                bandit_beta[r][ch]  = 1.0f;
            }
        float sum = 0.0f;
        for (int i = 0; i < HISTORY_WIN; ++i) {
            weights[i] = powf(DECAY, float(HISTORY_WIN - 1 - i));
            sum += weights[i];
        }
        for (int i = 0; i < HISTORY_WIN; ++i) weights[i] /= sum;
        score_weights[0][0] = 0.6f; score_weights[0][1] = 0.2f; score_weights[0][2] = 0.2f;
        score_weights[1][0] = 0.4f; score_weights[1][1] = 0.4f; score_weights[1][2] = 0.2f;
        score_weights[2][0] = 0.2f; score_weights[2][1] = 0.5f; score_weights[2][2] = 0.3f;
    }

    void estimate_weibull_ema(int ch, float x) {
        if (x < 1.0f) x = 1.0f;
        if (weibull_count[ch] == 0) {
            weibull_mean[ch] = x;
            weibull_var[ch]  = 0.0f;
        } else {
            float diff = x - weibull_mean[ch];
            weibull_mean[ch] += (1.0f - WEIBULL_EMA_ALPHA) * diff;
            weibull_var[ch] = WEIBULL_EMA_ALPHA * weibull_var[ch]
                            + (1.0f - WEIBULL_EMA_ALPHA) * diff * diff;
        }
        weibull_count[ch]++;
        int n = weibull_count[ch];
        if (n < WEIBULL_MIN_SAMPLES) return;
        float mean_t = weibull_mean[ch];
        float var_t  = weibull_var[ch];
        float std_t  = sqrtf(max(var_t, 0.0f));
        if (std_t < 1e-3f || mean_t < 1e-3f) {
            shape_k[ch]        = 1.0f;
            lambda_weibull[ch] = 1.0f / max(mean_t, 1.0f);
            weibull_valid[ch]  = true;
            return;
        }
        float cv = std_t / mean_t;
        float k, gamma_corr;
        WEIBULL_LUT.lookup(cv, k, gamma_corr);
        shape_k[ch]        = k;
        lambda_weibull[ch] = gamma_corr / max(mean_t, 1.0f);
        weibull_valid[ch]  = true;
    }

    inline float fast_weibull_pow(float val, float k) const {
        if      (k == 1.0f) return val;
        else if (k == 2.0f) return val * val;
        else if (k == 0.5f) return sqrtf(val);
        else if (k == 1.5f) return val * sqrtf(val);
        else if (k == 3.0f) return val * val * val;
        else                return powf(val, k);
    }

    float get_survival_score(int ch, int horizon) {
        if (horizon == 1 && cache_surv_valid) {
            return cache_surv_1[ch];
        }
        float lam = lambda_weibull[ch];
        float k   = shape_k[ch];
        float h   = float(horizon);
        float score;
        if (lam > 0 && k > 0 && h > 0 && weibull_valid[ch])
            score = expf(-fast_weibull_pow(lam * h, k));
        else
            score = 1.0f;
        if (horizon == 1)
            cache_surv_1[ch] = score;
        return score;
    }

    void meta_update(float recent_perf) {
        performance_history[perf_hist_idx] = recent_perf;
        perf_hist_idx = (perf_hist_idx + 1) % META_WINDOW;
        if (perf_hist_size < META_WINDOW) perf_hist_size++;
        if (perf_hist_size < 5) return;
        float recent_mean = 0.0f, older_mean = 0.0f;
        int half = min(perf_hist_size / 2, 5);
        for (int i = 0; i < half; ++i) {
            int idx     = (perf_hist_idx - 1 - i + META_WINDOW) % META_WINDOW;
            int old_idx = (perf_hist_idx - perf_hist_size + i + META_WINDOW) % META_WINDOW;
            recent_mean += performance_history[idx];
            older_mean  += performance_history[old_idx];
        }
        recent_mean /= float(half);
        older_mean  /= float(half);
        // If collisions increase, lower the threshold (be more conservative)
        if (recent_mean > older_mean)
            threshold_momentum = 0.9f * threshold_momentum - 0.01f;
        else
            threshold_momentum = 0.9f * threshold_momentum + 0.01f;
        threshold = clamp(threshold + threshold_momentum, 0.1f, 0.35f);
    }

    inline void observe_incremental(int t) {
        env.observe_single(t, ring_buffer, ring_head);
        ring_head = (ring_head + 1) % HISTORY_WIN;
    }

    inline float get_ring(int ch, int rel_idx) const {
        return ring_buffer[(ch << 2) + ((ring_head + rel_idx) & 3)];
    }

    inline void precompute_static_scores() {
        for (int ch = 0; ch < N_CHANNELS; ++ch) {
            pre_s_occ[ch]  = 1.0f - occ_cache[ch];
            pre_s_var[ch]  = 1.0f - min(var_cache[ch] * 4.0f, 1.0f);
            pre_s_surv_h1[ch] = (hops_done >= WARM_HOPS) ? get_survival_score(ch, 1) : 1.0f;
        }
        cache_surv_valid = true;
        pre_scores_valid = true;
    }

    inline int get_surv_interval(char regime) const {
        if (regime == 'A') return SURV_INTERVAL_A;
        if (regime == 'B') return SURV_INTERVAL_B;
        return SURV_INTERVAL_C;
    }

    void refresh_cache(int t, char regime) {
        int start_fill = (cache_t < 0) ? max(0, t - HISTORY_WIN) : cache_t + 1;
        for (int tt = start_fill; tt <= t; ++tt) {
            observe_incremental(tt);
            int head = (ring_head + 3) & 3;
            for (int ch = 0; ch < N_CHANNELS; ++ch) {
                int state = (ring_buffer[(ch << 2) + head] > 0.5f);
                if (last_state[ch] == 0) {
                    if (state == 0) current_idle[ch]++;
                    else {
                        estimate_weibull_ema(ch, (float)current_idle[ch]);
                        current_idle[ch] = 0;
                        last_state[ch] = 1;
                    }
                } else {
                    if (state == 0) {
                        current_idle[ch] = 1;
                        last_state[ch] = 0;
                    }
                }
            }
            cache_surv_valid = false;
        }

        int i0 = (ring_head + 0) & 3, i1 = (ring_head + 1) & 3;
        int i2 = (ring_head + 2) & 3, i3 = (ring_head + 3) & 3;
        float w0 = weights[0], w1 = weights[1], w2 = weights[2], w3 = weights[3];
        float alpha = IIR_ALPHA, oma = 1.0f - IIR_ALPHA;

        for (int ch = 0; ch < N_CHANNELS; ++ch) {
            const float* rb = &ring_buffer[ch << 2];
            float v0 = rb[i0], v1 = rb[i1], v2 = rb[i2], v3 = rb[i3];
            float occ_rate = v0*w0 + v1*w1 + v2*w2 + v3*w3;
            float sq_sum   = occ_rate; // Since v is 0 or 1, v*v == v

            if (cache_t < 0) occ_cache[ch] = occ_rate;
            else occ_cache[ch] = occ_cache[ch] * alpha + occ_rate * oma;
            var_cache[ch] = max(0.0f, sq_sum - occ_rate * occ_rate);
        }

        if (collision_ring_size >= 16) {
            int recent_coll = 0;
            for (int i = 0; i < collision_ring_size; ++i) recent_coll += collision_ring[i];
            float rc = float(recent_coll) / collision_ring_size;
            if      (rc > 0.15f) threshold = max(0.10f, threshold - 0.02f);
            else if (rc < 0.05f) threshold = min(0.35f, threshold + 0.01f);
        }

        if (t > 0 && t % 20 == 0) {
            float recent_perf = 0.0f;
            if (collision_ring_size > 0) {
                int total_coll = 0;
                for (int i = 0; i < collision_ring_size; ++i) total_coll += collision_ring[i];
                recent_perf = float(total_coll) / collision_ring_size;
            }
            meta_update(recent_perf);
        }

        n_safe = 0;
        for (int ch = 0; ch < N_CHANNELS; ++ch)
            if (occ_cache[ch] < threshold && excluded_ttl[ch] <= t)
                safe_channels[n_safe++] = ch;

        precompute_static_scores();
        cache_t = t;
    }

    inline void get_safe(int t, char regime) {
        if (t - cache_t >= CACHE_TTL || cache_t < 0)
            refresh_cache(t, regime);
        for (int ch = 0; ch < N_CHANNELS; ++ch)
            if (excluded_ttl[ch] > 0 && excluded_ttl[ch] <= t)
                excluded_ttl[ch] = 0;
    }

    inline pair<char, float> detect_regime(int t) {
        if (regime_cache_first || (hops_done & 1) == 0) {
            auto [free0, free1] = env.get_current_free(t);
            int n_free = __builtin_popcountll(free0) + __builtin_popcountll(free1);
            float ratio = float(n_free) / N_CHANNELS;
            ema_ratio = ema_alpha * ratio + (1.0f - ema_alpha) * ema_ratio;
            last_regime_ratio = ema_ratio;
            if      (ema_ratio > 0.50f) last_regime_char = 'A';
            else if (ema_ratio > 0.25f) last_regime_char = 'B';
            else                        last_regime_char = 'C';
            regime_cache_first = false;
        }
        return {last_regime_char, last_regime_ratio};
    }

    inline void compute_scores(int* channels, int n_ch, int horizon,
                               float* s_occ, float* s_surv, float* s_var) {
        if (horizon == 1 && pre_scores_valid) {
            for (int i = 0; i < n_ch; ++i) {
                int ch = channels[i];
                s_occ[i]  = pre_s_occ[ch];
                s_var[i]  = pre_s_var[ch];
                s_surv[i] = pre_s_surv_h1[ch];
            }
        } else {
            for (int i = 0; i < n_ch; ++i) {
                int ch = channels[i];
                s_occ[i]  = pre_s_occ[ch];
                s_var[i]  = pre_s_var[ch];
                s_surv[i] = (hops_done >= WARM_HOPS) ? get_survival_score(ch, horizon) : 1.0f;
            }
        }
    }

    int ucb1_select(int regime_idx) {
        if (n_safe == 0) { stat_fallback++; return (int)(rng() % N_CHANNELS); }
        flush_bandit_decay(regime_idx);
        float total_obs = 0.0f;
        for (int i = 0; i < n_safe; ++i)
            total_obs += (float)bandit_obs_count[regime_idx][safe_channels[i]];
        float log_total_2 = 2.0f * logf(max(total_obs, 1.0f));
        float best_ucb = -1e18f;
        int   best_ch  = safe_channels[0];
        for (int i = 0; i < n_safe; ++i) {
            int   ch   = safe_channels[i];
            int   n    = bandit_obs_count[regime_idx][ch];
            if (n == 0) { best_ch = ch; break; }
            float a    = bandit_alpha[regime_idx][ch];
            float b    = bandit_beta[regime_idx][ch];
            float ucb  = (a / (a + b)) + sqrtf(log_total_2 / (float)n);
            if (ucb > best_ucb) { best_ucb = ucb; best_ch = ch; }
        }
        total_bandit_calls++;
        return best_ch;
    }

    int mrv_select(int horizon, int regime_idx) {
        if (n_safe == 0) { stat_fallback++; return (int)(rng() % N_CHANNELS); }
        float w0 = score_weights[regime_idx][0], w1 = score_weights[regime_idx][1], w2 = score_weights[regime_idx][2];
        float best_score = -1e18f;
        int   best_ch = safe_channels[0];
        bool  use_h1 = (horizon == 1 && pre_scores_valid);
        bool  warm = (hops_done >= WARM_HOPS);
        for (int i = 0; i < n_safe; ++i) {
            int ch = safe_channels[i];
            float s_surv = use_h1 ? pre_s_surv_h1[ch] : (warm ? get_survival_score(ch, horizon) : 1.0f);
            float score = w0 * pre_s_occ[ch] + w1 * s_surv + w2 * pre_s_var[ch];
            if (score > best_score) { best_score = score; best_ch = ch; }
        }
        return best_ch;
    }

    int entropy_select() {
        if (n_safe == 0) { stat_fallback++; return int(rng()) % N_CHANNELS; }
        compute_scores(safe_channels, n_safe, 1, buf_s_occ, buf_s_surv, buf_s_var);
        int   prev       = hop_sequence.empty() ? N_CHANNELS / 2 : hop_sequence.back();
        float best_score = -1.0f;
        int   best_idx   = 0;
        for (int i = 0; i < n_safe; ++i) {
            float dist   = abs(safe_channels[i] - prev);
            float d_norm = dist / (float(N_CHANNELS) + 1e-6f);
            float score  = d_norm * (0.5f * buf_s_surv[i] + 0.3f * buf_s_occ[i] + 0.2f * buf_s_var[i]);
            if (score > best_score) { best_score = score; best_idx = i; }
        }
        return safe_channels[best_idx];
    }

    void gen_block(int bsize, int t_start, int regime_idx) {
        if (n_safe == 0) { block_size = 0; return; }
        if (bsize > n_safe) bsize = n_safe;
        if (bsize == 0)     { block_size = 0; return; }

        compute_scores(safe_channels, n_safe, bsize, buf_s_occ, buf_s_surv, buf_s_var);

        if (hops_done >= WARM_HOPS) {
            for (int i = 0; i < n_safe; ++i) {
                float ls = get_survival_score(safe_channels[i], bsize * 2);
                buf_s_occ[i] *= ls;
            }
        }

        for (int i = 0; i < n_safe; ++i)
            score_buffer[i] = score_weights[regime_idx][0] * buf_s_occ[i]
                            + score_weights[regime_idx][1] * buf_s_surv[i]
                            + score_weights[regime_idx][2] * buf_s_var[i];

        int need = min(bsize, n_safe);
        int top_idx[MAX_BLOCK];
        float top_score[MAX_BLOCK];
        int filled = 0;

        for (int i = 0; i < n_safe; ++i) {
            float sc = score_buffer[i];
            if (filled < need) {
                int j = filled - 1;
                while (j >= 0 && top_score[j] < sc) {
                    top_score[j + 1] = top_score[j];
                    top_idx[j + 1]   = top_idx[j];
                    j--;
                }
                top_score[j + 1] = sc;
                top_idx[j + 1]   = i;
                filled++;
            } else if (sc > top_score[need - 1]) {
                int j = need - 2;
                while (j >= 0 && top_score[j] < sc) {
                    top_score[j + 1] = top_score[j];
                    top_idx[j + 1]   = top_idx[j];
                    j--;
                }
                top_score[j + 1] = sc;
                top_idx[j + 1]   = i;
            }
        }

        int selected[MAX_BLOCK];
        for (int i = 0; i < need; ++i) selected[i] = safe_channels[top_idx[i]];

        bool used[MAX_BLOCK] = {false};
        current_block[0] = selected[0];
        used[0] = true;
        int seq_len = 1;
        while (seq_len < need) {
            int last     = current_block[seq_len - 1];
            int best_idx = -1, best_dist = -1;
            for (int i = 0; i < need; ++i) {
                if (used[i]) continue;
                int dist = abs(selected[i] - last);
                if (dist > best_dist) { best_dist = dist; best_idx = i; }
            }
            current_block[seq_len++] = selected[best_idx];
            used[best_idx] = true;
        }
        block_size = seq_len;
        block_idx  = 0;
    }

    void flush_bandit_decay(int ridx) {
        int pending = bandit_decay_pending[ridx];
        if (pending == 0) return;
        float decay = (ridx == 2) ? 0.88f : BANDIT_DECAY;
        float decay_n = powf(decay, float(pending));
        for (int c = 0; c < N_CHANNELS; ++c) {
            bandit_alpha[ridx][c] = 1.0f + decay_n * (bandit_alpha[ridx][c] - 1.0f);
            bandit_beta[ridx][c]  = 1.0f + decay_n * (bandit_beta[ridx][c]  - 1.0f);
        }
        bandit_decay_pending[ridx] = 0;
    }

    pair<int, bool> execute(int ch, int t, char regime) {
        bool coll = env.is_collision(ch, t);
        collision_ring[collision_ring_idx] = coll ? 1 : 0;
        collision_ring_idx = (collision_ring_idx + 1) % 32;
        if (collision_ring_size < 32) collision_ring_size++;

        int hidx = channel_hist_idx[ch];
        channel_collision_hist[ch][hidx] = coll ? 1 : 0;
        channel_hist_idx[ch]  = (hidx + 1) % 16;
        if (channel_hist_size[ch] < 16) channel_hist_size[ch]++;

        int ridx = (regime == 'A') ? 0 : (regime == 'B') ? 1 : 2;
        bandit_obs_count[ridx][ch]++;
        if (coll) bandit_beta[ridx][ch]  += 1.0f;
        else      bandit_alpha[ridx][ch] += 1.0f;

        bandit_decay_pending[ridx]++;
        if (bandit_decay_pending[ridx] >= BANDIT_DECAY_FLUSH)
            flush_bandit_decay(ridx);

        if (coll) {
            backtracks++;
            int recent_rate = 0;
            for (int i = 0; i < channel_hist_size[ch]; ++i)
                recent_rate += channel_collision_hist[ch][i];
            float rr = channel_hist_size[ch] > 0
                       ? float(recent_rate) / channel_hist_size[ch] : 0.5f;
            int dynamic_cd = int(base_cooldown * (1.0f + 5.0f * rr * rr));
            if      (regime == 'C') dynamic_cd = int(dynamic_cd * 1.5f);
            else if (regime == 'A') dynamic_cd = int(dynamic_cd * 0.8f);
            excluded_ttl[ch] = t + dynamic_cd;

            int best_alt = -1;
            float best_sc = -1.0f;
            for (int i = 0; i < n_safe; ++i) {
                int alt = safe_channels[i];
                if (alt == ch) continue;
                float sc = 1.0f - occ_cache[alt];
                if (sc > best_sc) { best_sc = sc; best_alt = alt; }
            }
            ch = (best_alt >= 0) ? best_alt : int(rng()) % N_CHANNELS;

            collisions++;
            hop_sequence.push_back(ch);
            return {ch, true};
        }
        hop_sequence.push_back(ch);
        return {ch, false};
    }


    inline bool need_block_regen(int bsize, char regime) {
        if (block_idx >= block_size) return true;
        if (block_idx < block_size / 2) return false;
        if (block_size <= 4) return true;
        if (collision_ring_size >= 8) {
            int recent_coll = 0;
            for (int i = 0; i < collision_ring_size; ++i) recent_coll += collision_ring[i];
            float rc = float(recent_coll) / collision_ring_size;
            return rc > 0.08f;
        }
        return true;
    }

    struct Result {
        int   collisions;
        float collision_rate;
        int   backtracks;
        int   stat_A, stat_B, stat_C, stat_fallback;
        int   total_hops;
        float final_threshold;
        double elapsed_ms;
        int   surv_skipped;
        int   bandit_calls;
    };

    Result run(int n_hops) {
        auto start = chrono::high_resolution_clock::now();
        int t = 0;
        while (t < n_hops) {
            auto [regime, ratio_free] = detect_regime(t);
            get_safe(t, regime);

            if (regime == 'A') {
                stat_A++;
                bool use_bandit = (ratio_free < 0.75f);
                if (use_bandit) {
                    int min_obs = INT_MAX;
                    for (int i = 0; i < n_safe; ++i) {
                        int ch = safe_channels[i];
                        if (bandit_obs_count[0][ch] < min_obs) min_obs = bandit_obs_count[0][ch];
                    }
                    if (min_obs < BANDIT_MIN_OBS) use_bandit = false;
                }
                int ch = use_bandit ? ucb1_select(0) : mrv_select(1, 0);
                auto [new_ch, hit] = execute(ch, t, 'A');
                t++;
            }
            else if (regime == 'B') {
                if (need_block_regen(4, regime)) {
                    int bsize = 4 + int((12 - 4) * ratio_free);
                    bsize = max(4, min(12, bsize));
                    bsize = min(bsize, n_safe);
                    gen_block(bsize, t, 1);
                    if (block_size == 0) {
                        stat_fallback++;
                        auto [new_ch, hit] = execute(mrv_select(1, 1), t, 'B');
                        t++; continue;
                    }
                    stat_B++;
                }
                int ch = current_block[block_idx++];
                bool ch_safe = false;
                for (int i = 0; i < n_safe; ++i)
                    if (safe_channels[i] == ch) { ch_safe = true; break; }
                if (!ch_safe) ch = mrv_select(1, 0);
                auto [new_ch, hit] = execute(ch, t, 'B');
                t++;
            }
            else {
                stat_C++;
                auto [new_ch, hit] = execute(entropy_select(), t, 'C');
                t++;
            }
            hops_done++;
        }
        for (int r = 0; r < 3; ++r) flush_bandit_decay(r);

        auto end = chrono::high_resolution_clock::now();
        double elapsed_ms = chrono::duration<double, milli>(end - start).count();
        return {
            collisions,
            float(collisions) / n_hops,
            backtracks,
            stat_A, stat_B, stat_C, stat_fallback,
            int(hop_sequence.size()),
            threshold,
            elapsed_ms,
            surv_skipped,
            total_bandit_calls
        };
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    const int N_HOPS = 2000;
    const int N_RUNS = 5000;
    const int WARMUP = 3;

    vector<double> times(N_RUNS);
    vector<float>  colls(N_RUNS);
    vector<float>  thresholds(N_RUNS);

    for (int w = 0; w < WARMUP; ++w) {
        int seed = 999 + w;
        SpectralEnvironment env(seed);
        SornalyV49 solver(env, seed);
        solver.run(N_HOPS);
    }

    for (int run = 0; run < N_RUNS; ++run) {
        int seed = 42 + run;
        SpectralEnvironment env(seed);
        SornalyV49 solver(env, seed);
        auto r = solver.run(N_HOPS);
        times[run] = r.elapsed_ms;
        colls[run] = r.collision_rate;
        thresholds[run] = r.final_threshold;
    }

    double avg_time = 0.0;
    float avg_coll = 0.0, avg_thresh = 0.0;
    double min_time = 1e9, max_time = 0;
    float min_coll = 1.0f, max_coll = 0.0f;
    for (int i = 0; i < N_RUNS; ++i) {
        avg_time += times[i]; avg_coll += colls[i]; avg_thresh += thresholds[i];
        min_time = min(min_time, times[i]); max_time = max(max_time, times[i]);
        min_coll = min(min_coll, colls[i]); max_coll = max(max_coll, colls[i]);
    }
    avg_time /= N_RUNS; avg_coll /= N_RUNS; avg_thresh /= N_RUNS;

    double var_time = 0;
    float var_coll = 0;
    for (int i = 0; i < N_RUNS; ++i) {
        var_time += (times[i] - avg_time) * (times[i] - avg_time);
        var_coll += (colls[i] - avg_coll) * (colls[i] - avg_coll);
    }
    double std_time = sqrt(var_time / N_RUNS);
    float std_coll = sqrt(var_coll / N_RUNS);

    cout << "=== SORNALY V49 (CORRIGE – base V47, pas de regression) ===" << endl;
    cout << "Runs: " << N_RUNS << " (warmup: " << WARMUP << "), Hops/run: " << N_HOPS << endl;
    cout << "Avg collision rate: " << fixed << setprecision(3) << avg_coll*100 << "%" << endl;
    cout << "Std collision rate: " << fixed << setprecision(3) << std_coll*100 << "%" << endl;
    cout << "Min/Max collision:  " << fixed << setprecision(3) << min_coll*100 << "% / " << max_coll*100 << "%" << endl;
    cout << "Avg time: " << avg_time << " ms" << endl;
    cout << "Std time: " << std_time << " ms" << endl;
    cout << "Min/Max time: " << min_time << " / " << max_time << " ms" << endl;
    cout << "Avg threshold: " << avg_thresh << endl;
    return 0;
}
