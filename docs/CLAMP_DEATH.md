# MatUp 채널 사망: clamp가 분할 MLP를 불안정하게 만드는 이유

`matup_shared_vs_split/ep10`의 분할 arm에서 일부 seed가 엉뚱한 결과로 끝났습니다(plastic 25.5 dB, roofing −3.7 dB 등). 이 문서는 그 원인 가설을 정리하고, 가설을 가리는 실험을 정의합니다.

실험은 `scripts/matup_split.sh`의 `probe`, `report` 단계와 `*_zinit`, `*_noreg` arm으로 돌립니다. 결과는 §6, 시도 후 뺀 straight-through clamp(`*_st`)는 §7에 있습니다. 명령은 [COMMANDS.txt](../COMMANDS.txt) [5-4]~[5-6]에 있습니다. 공유·분할 MLP의 정의는 [SPLIT_THEORY.md](SPLIT_THEORY.md) B1을 봅니다.

표시: **[관측]** 몽타주 맵이나 코드에서 직접 확인한 것, **[추정]** 분포를 가정해 계산하거나 추론한 것.

---

## 1. 관측

ep10 몽타주의 맵 픽셀을 읽으면, 결과가 몇 개의 값으로 나뉘는 이유가 **어느 채널이 clamp 경계에 붙었는지**와 정확히 대응합니다 **[관측]**.

plastic_009의 baseColor입니다. GT 평균은 sRGB (13, 43, 18)입니다.

| 경계(0)에 붙은 albedo 채널 | run | render PSNR (train100) |
|---|---|---|
| 없음 | split $s_1$ | 40.15 dB |
| $R$ | split $s_2$, x1 $s_1$, x3 $s_0$, x3 $s_2$ | 34.5 dB |
| $R, B$ | **shared $s_0, s_1, s_2$**, x1 $s_2$, x3 $s_1$ | 31.8 dB |
| $R, G, B$ (검정) | split $s_0$, x1 $s_0$ | 25.5 dB |

다른 재질도 같은 패턴입니다.

- **roofing_tiles_006 (x1 $s_2$, x3 $s_1$):** $B = 0$이라 주황색으로 보이고, render는 −3.7 dB입니다.
- **brick_wall_005 (split $s_2$):** roughness 맵이 모든 texel에서 255입니다. $\alpha$가 위쪽 경계 1에 붙은 경우이고, render에는 거의 영향이 없습니다.
- **syntetic_foam (split 계열 전부):** roughness 맵이 완전히 상수(std 0)입니다. 다만 경계가 아니라 정답 근처의 값이라 오히려 +10 dB입니다.
- **폭과 무관함:** 폭이 다른 arm(104, 128, 227)도 같은 상태에 빠집니다. 죽은 채널은 정확히 경계값이라 맵이 같아지고, 그래서 PSNR이 소수점까지 같습니다.

---

## 2. 모형

### 2.1 순전파

| 기호 | 뜻 |
|---|---|
| $i = 1,\dots,N$ | texel |
| $c$ | 출력 채널. albedo $R,G,B$ / metallic $m$ / normal $n_x,n_y$ / alpha $\alpha$ |
| $x_i$ | texel $i$ 주변 $17\times17$ 타일의 bilinear 초기 맵 (MLP 입력) |
| $s_{i,c}$ | 중심 초기값 (`svbrdf_mid_sample`) |
| $f_\theta$ | MLP. 마지막 층은 $f_c(x) = w_c^{\top}h(x) + b_c$, $h\in\mathbb R^{W}$ |
| $[l_c,u_c]$ | clamp 범위 (`get_renorm_clamped_maps`). albedo·$m$은 $[0,1]$, $\alpha$는 $[\alpha_{\min},1]$, $\alpha_{\min}=2\cdot10^{-3}$ |

$$
z_{i,c} = s_{i,c} + f_{\theta,c}(x_i),\qquad
y_{i,c} = \Pi_c(z_{i,c}) = \min\!\big(\max(z_{i,c},l_c),u_c\big)
$$

$$
\mathcal L = \frac{1}{NK}\sum_{i}\sum_{k=1}^{K}\big\lVert R(y_i,\omega_k)-T_{i,k}\big\rVert_2
\;+\;\lambda_{\mathrm{reg}}\frac1N\sum_i\big(y_{i,m}-s_{i,m}\big)^2,
\qquad K=100,\ \lambda_{\mathrm{reg}}=0.1
$$

### 2.2 active set과 채널 사망

clamp의 미분은 범위 안에서 1, 밖에서 0입니다. PyTorch에서는 경계값도 범위 안으로 칩니다.

$$
\frac{\partial y_{i,c}}{\partial z_{i,c}} = \mathbb 1\!\left[l_c\le z_{i,c}\le u_c\right],
\qquad
\mathcal A_c = \{\,i : l_c\le z_{i,c}\le u_c\,\}
$$

$g_{i,c}:=\partial\mathcal L/\partial y_{i,c}$로 두면 gradient는 active set 위의 합으로만 남습니다.

$$
\nabla_\theta\mathcal L = \sum_c\sum_{i\in\mathcal A_c} g_{i,c}\,\nabla_\theta f_c(x_i)
$$

metallic 정규화항도 clamp 뒤의 $y$에 걸려 있으므로 같은 지시함수가 곱해집니다. 이제 **채널 사망**을 $\mathcal A_c=\varnothing$로 정의합니다.

$$
\mathcal A_c=\varnothing
\;\Longrightarrow\;
\frac{\partial\mathcal L}{\partial w_c}=\sum_{i\in\mathcal A_c}g_{i,c}\,h(x_i)=0,
\qquad
\frac{\partial\mathcal L}{\partial b_c}=0
$$

죽은 뒤에도 Adam은 남은 momentum으로 같은 방향으로 더 움직입니다(MatUp은 $\eta=10^{-4}$, $\beta_1=0.9$).

$$
\sum_{k\ge0}\Delta\theta_{t+k}\approx-\frac{\eta}{1-\beta_1}\frac{m_t}{\sqrt{v_t}} = -10\,\eta\,\frac{m_t}{\sqrt{v_t}}
$$

약 10스텝을 더 가서 경계 밖으로 더 밀려납니다.

### 2.3 부활: shared와 split의 차이

$(w_c,b_c)$가 멈춰도 은닉층 $h$가 바뀌면 $z$는 움직입니다.

$$
\Delta z_{i,c} = w_c^{\top}\Delta h(x_i),
\qquad
\nabla_{\theta_h}\mathcal L=\sum_{c'}\sum_{i\in\mathcal A_{c'}}g_{i,c'}\,J_h(x_i)^{\top}w_{c'}
$$

- **shared:** $c'$는 7채널 전부입니다. albedo가 다 죽어도 normal, $m$, $\alpha$ 항이 $h$를 계속 움직입니다. 그래서 다시 $\mathcal A_c\neq\varnothing$가 될 수 있습니다.
- **split:** 그룹 $g$의 MLP는 자기 그룹 채널의 gradient만 받습니다. 그룹 채널이 모두 죽으면 그 MLP 전체가 영구히 멈춥니다(흡수 상태).

$$
\bigcap_{c\in g}\{\mathcal A_c=\varnothing\}
\;\Longrightarrow\;
\nabla_{\theta^{g}}\mathcal L=0
\;\Longrightarrow\;
\Delta z_{\cdot,c}\equiv0\quad(c\in g)
$$

plastic split $s_0$의 검정 baseColor가 이 상태로 보입니다. rm-MLP는 채널이 $\{m,\alpha\}$ 둘뿐이라 더 쉽게 빠집니다.

### 2.4 공간적 퍼짐 $\sigma_c$

출력을 texel 공통 부분과 texel별 편차로 나눕니다.

$$
z_{i,c}=s_{i,c}+\mu_c+\varepsilon_{i,c},
\qquad
\mu_c=\tfrac1N\textstyle\sum_i f_c(x_i),
\qquad
\sigma_c=\operatorname{std}_i\varepsilon_{i,c}
$$

아래쪽 경계에서 사망 조건은 다음과 같습니다($\kappa\approx2$–$3$은 $N$개 texel 중 최댓값 효과).

$$
\mathcal A_c=\varnothing
\iff
\max_i\big(s_{i,c}+\varepsilon_{i,c}\big)\le-\mu_c
\;\approx\;
\mu_c\le-\big(\bar s_c+\kappa\,\sigma_c\big)
$$

$\sigma_c$는 입력이 얼마나 흔들리는지에 비례합니다.

$$
\varepsilon_{i,c}\approx\nabla_x f_c(\bar x)^{\top}(x_i-\bar x)
\;\Rightarrow\;
\sigma_c\approx\lVert\nabla_x f_c\rVert\cdot\operatorname{std}(x)
$$

| | 입력 | $\operatorname{std}(x)$ | $\sigma_c$ |
|---|---|---|---|
| shared | albedo, $m$, $n_{xy}$, $\alpha$ | normal 무늬 때문에 큼 | 큼 |
| split a-MLP, plastic | 어두운 단색 albedo만 | $\approx0$ | $\approx0$ |
| split rm-MLP, foam | 평평한 $m,\alpha$만 | $0$ | $0$ **[관측: roughness std 0]** |

입력이 거의 0이면 $W_1x+b_1\approx b_1$이라 $f_c(x_i)\approx f_c(0)$이 됩니다. 이때 출력은 bias가 정하는 상수 하나이고, 사망 조건은 $\mu_c\le-\bar s_c$로 단순해집니다. 한 번 넘으면 **모든 texel이 동시에** 넘습니다.

### 2.5 여유 $\bar s_c$와 초기 $\mu_c(0)$

MatUp은 albedo를 linear로 씁니다(`srgb2linrgb`). 여유는 다음과 같습니다 **[관측]**.

$$
\bar s=\Big(\tfrac{v/255+0.055}{1.055}\Big)^{2.4}
$$

| 채널 | $v$ (sRGB) | $\bar s$ | ep10에서 죽은 run |
|---|---|---|---|
| plastic $R$ | 13 | 0.004 | 11/12 |
| plastic $B$ | 18 | 0.006 | 7/12 |
| plastic $G$ | 43 | 0.024 | 2/12 (split 계열만) |
| roofing $B$ | 45 | 0.026 | 2/12 (split 계열만) |
| mud_leaves $B$ | 53 | 0.035 | 0/12 |

PyTorch `Conv2d`의 기본 초기화에서는 bias가 $\mathcal U(\pm1/\sqrt{\text{fan\_in}})$이고, 마지막 층의 fan_in은 $W$입니다 **[추정]**.

$$
\sigma_\mu:=\operatorname{std}\big(\mu_c(0)\big)\ \ge\ \frac{1}{\sqrt{3W}}
\qquad(W=104:\ 0.057,\quad 128:\ 0.051,\quad 227:\ 0.038)
$$

$\mu_c(0)\sim\mathcal N(0,\sigma_\mu^2)$로 가정하고 $\sigma_c\approx0$이면 초기화 순간에 죽을 확률은 다음과 같습니다.

$$
P_{\text{init}}(c)=\Phi\!\left(-\frac{\bar s_c}{\sigma_\mu}\right):
\quad R,\,W{=}128:\ 0.47,\qquad R,\,W{=}227:\ 0.46,\qquad G,\,W{=}128:\ 0.32
$$

폭을 2.2배로 늘려도 확률은 거의 그대로입니다. 크게 줄이려면 $W\gg 1/(3\bar s^2)\approx2\times10^{4}$이 필요합니다. x3로 붕괴가 사라지지 않은 이유입니다.

### 2.6 학습 중 drift

$$
\mu_c(t)=\mu_c(0)+d_c(t),
\qquad
\text{최종 사망}\iff\mu_c(0)+d_c\le-\big(\bar s_c+\kappa\sigma_c\big)
$$

$$
\text{seed와 무관한 결과}\iff|d_c|\gg\sigma_\mu,
\qquad
\text{seed 복권}\iff|d_c|\lesssim\sigma_\mu
$$

- **shared:** albedo 출력은 $h$를 공유합니다. $h$는 normal과 roughness의 큰 gradient로 움직이므로 $|d_c|$가 크고 방향이 seed와 무관합니다. 세 seed 모두 $R,B$가 죽고 $G$는 산 것과 맞습니다 **[관측]**. 초기 사망만으로 이 결과가 나올 확률은 $(0.47\cdot0.46)^3\approx0.01$이라, shared의 사망은 drift 때문으로 봅니다 **[추정]**.
- **split a-MLP:** albedo gradient만 받습니다. 어두운 diffuse는 렌더 미분 $\partial r/\partial a_c=(1-m)E\cos\theta/\pi$와 잔차가 모두 작아서 $|d_c|$가 작습니다. 그러면 seed가 결과를 정하고, 0~3채널 사망의 네 상태로 흩어집니다 **[관측]**.
  - B는 split 계열에서 9개 중 5개가 살아 shared(3개 중 0개)보다 나았습니다. 분할이 평균을 바꾼 것이 아니라 결과를 결정적인 것에서 무작위로 바꿨다는 뜻입니다.
  - 그래서 seed 편차가 shared $\pm0.01$에서 split $\pm0.62$ dB로 커졌습니다.

### 2.7 위쪽 경계

$$
\mathcal A_\alpha=\varnothing\iff\mu_\alpha\ \gtrsim\ (1-\bar s_\alpha)+\kappa\sigma_\alpha
$$

brick_wall은 LR roughness가 평평해 $\sigma_\alpha\approx0$이고, 모든 texel이 $\alpha=1$에서 멈췄습니다 **[관측]**.

### 2.8 metallic 정규화항은 주원인이 아님 [추정]

- **albedo에는 직접 닿지 않습니다.** 분할에서 albedo MLP는 $z_m$에 닿지 않으므로 $\nabla_{\varphi_a}\mathcal L_{\mathrm{reg}}\equiv0$입니다. 렌더를 통한 간접 효과는 $(1-m)$ 배율 3% 이내입니다.
- **크기가 작습니다.** texel당 gradient가 $2\lambda_{\mathrm{reg}}|y_m-s_m|/N\approx0.006/N$로 렌더 gradient의 1~5%입니다. 또 볼록한 복원항이라 스스로 발산하지 않습니다.
- **rm-MLP에서는 기여할 수 있습니다.** rm-MLP 안에서는 $\{m,\alpha\}$ 둘만 $h^{rm}$을 움직이므로 정규화항의 상대 비중이 커집니다. brick_wall의 $\alpha=1$ 고정에 기여했을 수 있습니다.

---

## 3. 수정안

**I1. 마지막 층 0 초기화 (`--zero-init`)**

$$
w_c=0,\ b_c=0\ \Rightarrow\ f_c\equiv0\ \text{at}\ t=0\ \Rightarrow\ \mathcal A_c=\{1..N\},\quad P_{\text{init}}=0
$$

초기 사망은 없애지만 drift에 의한 사망은 남습니다. 은닉층 gradient는 첫 스텝에서 0이지만($\partial f/\partial h=w=0$), 마지막 층이 바로 학습되면서 풀립니다. 생성 후에 0으로 바꾸므로 seed의 난수 흐름은 같습니다.

**I2. 학습 중 straight-through clamp (`--st-clamp`) — 시도 후 제외 (§7)**

$$
\text{forward: } y=\Pi(z),\qquad
\text{backward: } \frac{\partial y}{\partial z}:=1
\quad\Rightarrow\quad
\nabla_\theta\mathcal L=\sum_c\sum_{i}g_{i,c}\nabla_\theta f_c(x_i)
$$

gradient가 0이 되지 않으므로 흡수 상태가 없습니다. 렌더가 더 큰 값을 원하면($g_{i,c}<0$) $z$가 경계 안으로 돌아옵니다. 출력(forward)은 MatUp과 같습니다.

두 수정 모두 shared와 split에 똑같이 적용해 짝을 지어 비교합니다(`split_<v> − shared_<v>`).

---

## 4. 가설과 예측

| | 가설 | 측정 (§5) | 예측 |
|---|---|---|---|
| H1 | 경계에 붙은 맵 = active set이 빈 채널 | probe의 `active` | §1의 상수 채널은 마지막 probe에서 `active = 0` |
| H2 | split은 출력의 공간 퍼짐이 작음 | 초기 $\sigma_c$ (`d_std`, tag 0) | plastic albedo, foam·brick $\alpha$에서 split $\sigma\ll$ shared |
| H3 | split 그룹이 모두 죽으면 그 MLP가 멈춤 (흡수) | 그룹 `param_step`, 부활 수 | split의 전체 사망 그룹은 step ≈ 0이고 부활 없음. shared에는 부활이 있을 수 있음 |
| H4 | shared 사망은 drift, split 사망은 init과 drift가 섞임 | 처음 죽은 tag | shared $R,B$는 tag ≥ 1에서 죽음. split은 tag 0 사망이 상당수 |
| H5a | 0 초기화는 초기 사망을 없앰 | `*_zinit` | tag 0 사망 = 0. 남는 사망은 drift |
| H5b | ST clamp는 흡수 상태를 없앰 | `*_st` | 마지막에 죽은 채널이 거의 없음. plastic seed 편차 축소. split − shared 차이에서 clamp 효과가 빠짐 |
| H6 | metallic 정규화는 albedo 사망과 무관 | `*_noreg` vs 기존 (같은 seed) | plastic, roofing의 albedo 사망 상태 그대로. brick $\alpha$는 바뀔 수 있음 |

예측과 반대로 나오는 경우의 해석은 다음과 같습니다.

- **H2가 틀린 경우** (split의 초기 $\sigma$가 shared와 비슷): "출력이 균일해져 한 번에 죽는다"는 설명이 틀렸고, 다른 메커니즘을 찾아야 합니다.
- **H3이 틀린 경우** (전체 사망 그룹이 계속 움직임): 흡수 상태 설명이 틀렸습니다.
- **H5b가 틀린 경우** (ST 후에도 붕괴): clamp가 아니라 다른 최적화 문제입니다.
- **H6이 틀린 경우** (noreg에서 albedo 사망이 사라짐): 정규화항이 렌더를 통해 간접적으로 크게 작용한다는 뜻입니다.

---

## 5. 실험

모든 실험은 ep10과 같은 재질 12개, seed 0~2(1996~1998), 같은 SwinIR 캐시를 씁니다.

| 실험 | 내용 | 명령 | 출력 | 시간 |
|---|---|---|---|---|
| E0 | 정규화 끄기: `matup_shared_noreg`, `matup_split_noreg` | `optim matup_shared_noreg,matup_split_noreg` | 평가 arm | 72 run, ~45분 |
| E1 | probe: shared, split을 계측과 함께 다시 실행 | `probe` | `runs/matup_split/probe/`, 평가 arm 아님 | 72 run, ~50분 |
| E2 | 수정안: `*_zinit` (probe 포함; `*_st`는 §7에서 제외) | `optim matup_shared_zinit,matup_split_zinit` | 평가 arm | 72 run, ~45분 |
| 보고 | probe 표 | `report` | `runs/eval/matup_clamp_probe/ep10/` | 몇 초 |

**probe가 기록하는 것** (`variants/matup_split/clamp_probe.py`): 초기화 직후(tag 0)와 매 epoch 끝(tag 1~10)에 전체 재질의 $z=s+f(x)$를 계산하고, clamp가 걸리는 채널($R,G,B,m,\alpha$)마다 다음을 저장합니다.

- `active`, `below`, `above`
- `s_mean` ($\bar s$), `d_mean` ($\mu$), `d_std` ($\sigma$), `z_min`, `z_max`
- 마지막 층 행의 변화량 `row_step`
- 그룹별 파라미터 변화량 `param_step` (split: a, n, rm / shared: 전체)

MatUp의 epoch별 `eval_model`(`--save_interval 1`)을 그대로 쓰고, epoch별 PNG는 쓰지 않습니다. 학습은 바뀌지 않지만 GPU 연산이 비결정적이라 E1의 최종 상태는 ep10과 조금 다를 수 있습니다. 보고서에는 probe 재실행 자체의 상태를 씁니다.

**판정** (`probe_report.py`의 표)

| 가설 | 보는 표 |
|---|---|
| H1, H4 | 모든 사망 목록 (처음 죽은 tag, 끝 상태) |
| H2 | 초기 $\sigma$ 표 (재질 × shared/split) |
| H3 | 동결 그룹 표, arm별 부활 수 |
| H5 | arm별 요약 (`*_zinit`) |
| H6 | E0의 summary.md와 몽타주 (채널 사망은 맵 픽셀로 확인) |

E0과 E2는 평가 arm이라 `bash scripts/eval.sh score matup/ep10`, `bash scripts/eval.sh montage matup/ep10 --seeds 0,1,2`로 render 지표와 짝 비교가 함께 나옵니다.

---

## 6. 결과 (probe, ep10)

표는 `runs/eval/matup_clamp_probe/ep10/summary.md`에 있습니다. 재질 12개 × seed 3개 = arm당 36 run입니다. GPU 연산이 비결정적이라 probe 재실행은 ep10과 run 단위로 같지 않습니다. 그래서 개별 seed끼리 비교하지 않고 개수와 비율로 봅니다 **[관측]**.

| arm | 끝에 죽은 채널 | 그중 초기(tag 0) 사망 | drift 사망 | 전체가 죽은 그룹 |
|---|---|---|---|---|
| shared (probe 재실행) | 6 | 5 | 1 | 0 |
| split (probe 재실행) | 4 | 2 | 2 | 1 |
| shared_zinit | 0 | 0 | 0 | 0 |
| split_zinit | 0 | 0 | 0 | 0 |

| 가설 | 판정 | 근거 |
|---|---|---|
| H1 | 맞음 | plastic의 경계에 붙은 $R,B$는 끝 probe에서 `active = 0` |
| H2 | 약함 | 초기 $\sigma_0\le0.002$이고 albedo의 $\lvert\mu_0\rvert\approx0.058$은 재질과 arm에 상관없이 같습니다. 초기 사망은 $\sigma$가 아니라 $\mu_0$ 대 여유 $\bar s$로 갈립니다 |
| H3 | 맞음 | split plastic $s_0$의 a 그룹은 9 epoch 동안 전부 죽어 있었고 `param_step` = 0.00 (살아 있을 때 중앙값 1.55). 재실행 shared의 부활도 0 |
| H4 | 틀림 | shared 사망 6개 중 5개가 tag 0 (초기 사망) |
| H5a | 맞음 (예측보다 강함) | zinit은 초기 사망뿐 아니라 drift 사망도 0 |
| H5b | 틀림 → 제외 | §7 |
| H6 | 평가 대기 | `matup_shared_vs_split/ep10/summary.md` |

**결론.** 채널 사망은 공유와 분할 모두에서 일어납니다. 원인은 주로 초기화입니다. 무작위 마지막 층이 모든 texel에 거의 같은 오프셋 $\mu_0\approx\pm0.06$을 더합니다. 어두운 채널은 여유가 그보다 작아(plastic $R$ 0.0034, $B$ 0.0062, $G$ 0.0224) 음의 $\mu_0$가 나오면 바로 경계 밖으로 나갑니다. 분할은 죽은 그룹 MLP가 멈추므로 되살아나지 못합니다. 0 초기화는 이것을 없앱니다. 그러므로 공유 대 분할의 공정한 비교는 `matup_split_zinit − matup_shared_zinit`입니다. 붕괴 자체는 `montage/zinit/`(위: 원래 MatUp, 아래: zinit)에서 볼 수 있습니다.

## 7. straight-through clamp(`*_st`)를 뺀 이유

**(1) 첫 구현의 float32 소거 (0ec17d0에서 수정).** `r + (clamp(r) - r).detach()`는 $\lvert r\rvert>2^{24}$이면 clamp가 사라집니다. 예를 들어 $10^8+(1-10^8)=0$입니다. 평가 self-check가 1(전체 범위) 차이로 멈춘 원인이었습니다. 고친 forward `clamp(r).detach() + (r - r.detach())`는 정확히 $\Pi(r)$입니다.

**(2) 고친 뒤에도 경계 밖으로 폭주합니다 [관측 + 추정].**

| arm | 끝에 죽은 채널 | drift 사망 | 전체가 죽은 그룹 |
|---|---|---|---|
| shared_st | 6 | 6 | 3 |
| split_st | 18 | 17 | 14 |

split_st의 사망은 대부분 $\alpha$입니다. brick_uneven, cobblestones, roofing, brick_wall에서 모든 seed가 epoch 1~3에 죽었습니다. 이 재질들은 원래 MatUp에서는 $\alpha$가 죽지 않습니다. rm 그룹은 "전부 죽은" 상태인데도 `param_step`이 $3\times10^1$~$1.4\times10^2$로, 살아 있을 때(~7)보다 훨씬 큽니다. 멈춘 것이 아니라 달려 나가는 중입니다.

메커니즘은 이렇습니다. 경계에 닿은 texel에서 렌더가 경계 너머를 원하면 $g=\partial\mathcal L/\partial y$는 바깥을 가리킵니다.
- **hard clamp:** 이 gradient를 0으로 끊습니다. 그 texel은 경계에 머물고, 나머지 texel은 계속 맞춰집니다.
- **ST:** 이 gradient를 그대로 통과시킵니다. 공유 가중치를 통해 모든 texel을 바깥으로 끕니다.

출력은 clamp되어 손실이 변하지 않습니다. 그래서 gradient도 같은 값으로 계속 나오고, Adam은 매 스텝 거의 lr만큼 같은 방향으로 움직입니다. 이 과정을 멈출 것이 없습니다. split의 rm MLP는 $m,\alpha$의 gradient만 받으므로 이 끌림이 희석되지 않습니다. shared는 trunk가 albedo와 normal의 gradient도 받아 덜합니다.

결과적으로 brick split_st의 train100 $\ell_{2,1}$은 0.044입니다. bilinear(0.038)와 다른 arm(~0.042)보다 나쁩니다. plastic의 사망은 shared_st에서 되살아났지만(alive 1.00), 그 대가로 다른 재질이 망가졌습니다.

**(3) 평가 self-check가 넘칩니다.** ST run만 MatUp의 `render_opt.png`와 16-bit `_opt` PNG에서 다시 그린 렌더의 차이가 허용치 5e-4를 넘었습니다.
- brick split_st: 1.2e-3, 픽셀 값의 0.04%
- foam shared_st $s_0$: 1.2e-2

다른 arm은 같은 코드에서 모두 ≤ 12 LSB(≈ 1.8e-4)라 맵 준비 버그는 아닙니다. 극단값 맵에서 16-bit 왕복 오차가 커진 것으로 보입니다 **[추정]**. 다만 foam의 정확한 메커니즘은 찾지 못했습니다($\alpha=1$ 고정은 민감한 값이 아닙니다). ST를 빼면서 더 조사하지 않았습니다.

**(4) 부수 문제.** 실패한 score 실행이 이전에 성공한 `matup_shared_vs_split/ep10/{report.json,summary.md}`를 지운 채로 커밋됐습니다(1368159). ST를 뺀 뒤 score를 다시 돌려 복구합니다.
