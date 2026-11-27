# 맵 그룹 분할 네트워크: 기본 논문과 달라지는 점

- 작성: 2026-11-23
- 이전 문서: comp2 `docs/DEEPTEX_SPLIT_THEORY.md`(2026-11-21). 이 문서가 그 문서를 대체합니다.
  - 학습 방식: 이전 문서의 D1(나머지 맵을 GT로 채워 독립 학습)이 아니라 **공동 학습**입니다.
  - 파라미터: 1/3로 맞추지 않고 nb=23(3배)과 nb=7(약 0.93배)을 둘 다 돌립니다.
  - MatUp: SwinIR-L이 아니라 재질별 MLP를 분할합니다.
- 교수님 답변(2026-11-23): "파라미터 개수는 3배로 늘어나도 됩니다. 어차피 나누어져 따로 추론하는 모델이니까."

**가설:** 다른 맵의 정보를 보지 않는 맵 그룹 전용 네트워크가, 5맵을 공유하는 네트워크보다 맵 품질과 렌더 품질에서 더 낫다.

---

## 0. 기호

| 기호 | 의미 |
|---|---|
| $X=(A,N,R,M,H)$ | 재질: albedo, normal, roughness, metallic, height |
| $x=\mathcal D_{\downarrow4}(X)$ | LR 입력 (area 다운샘플) |
| $\bar R=[R,R,R]$ | 회색 맵을 3채널로 복제한 이미지 (GGG) |
| $\rho(X;p,\omega_l)$ | texel $p$를 빛 $\omega_l$ 아래 렌더한 값. 빛은 $L=100$개 (MatUp의 학습 빛) |
| $V=\{p:\lVert2N_p-1\rVert\ge0.1\}$ | GT normal이 유효한 texel |
| $g\in\{a,n,rm\}$ | 맵 그룹: albedo / normal / roughness·metallic |

render loss (MatUp의 $\ell_{2,1}$):

$$
\mathcal L_{\text{ren}}(\hat X,X)=\frac{1}{|V|L}\sum_{p\in V}\sum_{l=1}^{L}\big\lVert\rho(\hat X;p,\omega_l)-\rho(X;p,\omega_l)\big\rVert_2
$$

$\rho(X)$(GT 쪽)는 tape 밖에서 한 번만 계산하므로 gradient가 흐르지 않습니다.

---

## A. DeepTextures

### A0. 기본 논문 (DeepTextures = ESRGAN 레시피를 텍스처 맵에 적용)

G 하나가 **맵 종류와 상관없이** 이미지 한 장씩 SR합니다. 배치는 아무 맵이나 섞은 8장입니다.

$$
\mathcal L_G^{\text{DT}}=\frac1{8}\sum_{i=1}^{8}\Big[w_{\text{pix}}\lVert G_\theta(x_i)-y_i\rVert_1+w_{\text{feat}}\lVert\phi(G_\theta(x_i))-\phi(y_i)\rVert_1\Big]+w_{\text{gan}}\,\mathcal L_{\text{RaGAN}}(D_\psi)
$$

- $w_{\text{pix}}=10^{-2}$, $w_{\text{feat}}=1$ (VGG54), $w_{\text{gan}}=5\cdot10^{-3}$ (RaGAN).
- RRDB는 nb=23, nf=64, gc=32입니다. D는 VGG128 하나입니다.
- render loss는 없습니다.

### A1. comp2 `deeptex_renloss_21` (기본 논문에서 이미 바뀐 것)

render loss를 쓰려면 같은 재질의 5맵이 같은 crop으로 함께 있어야 합니다. 그래서 데이터 단위를 재질로 바꿨습니다(grouped TFRecord). 배치는 재질 $B=2$개 × 5맵 = 이미지 10장입니다(원래 레시피의 8장에 맞춘 것).

$$
\mathcal L_G^{\text{shared}}=\frac1{5B}\sum_{b=1}^{B}\sum_{m\in\{A,N,R,M,H\}}\Big[w_{\text{pix}}\lVert G_\theta(x_{b,m})-y_{b,m}\rVert_1+w_{\text{feat}}\lVert\phi(G_\theta(x_{b,m}))-\phi(y_{b,m})\rVert_1\Big]+w_{\text{gan}}\,\mathcal L_{\text{RaGAN}}(D_\psi)+w\,\mathcal L_{\text{ren}}(\hat X,X)
$$

$$
\hat X=\big(G_\theta(a),\ G_\theta(n),\ \mathrm{mean}_c\,G_\theta(\bar r),\ \mathrm{mean}_c\,G_\theta(\bar m),\ \mathrm{mean}_c\,G_\theta(\bar h)\big),\qquad w=10
$$

- $w=10$은 comp1 calibration gate(2026-10-07)에서 정한 값입니다.

### A2. comp3 분할 (공동 학습)

**① 네트워크가 3개로 나뉩니다.**

$$
\hat y_a=G_{\theta_a}(a),\qquad \hat y_n=G_{\theta_n}(n),\qquad \hat y_{rm}=G_{\theta_{rm}}([0,\,r,\,m])
$$

공유에서는 $\theta$ 하나가 5맵을 모두 처리했습니다. 이제는 그룹마다 $\theta_g$가 따로 있습니다.

| | 공유 (nb=23) | 분할 nb=23 ×3 | 분할 nb=7 ×3 |
|---|---|---|---|
| G 파라미터 | 16,697,987 | 50,093,961 (3배) | 15,561,609 (0.93배) |
| 이미지당 연산 | 1 | 1 | 약 0.36 |

RRDB 파라미터 수는 $P_G(n_b)=151{,}235+719{,}424\,n_b$입니다.

**② RM은 묶어서 넣고, 채널을 골라 꺼냅니다.**
- 공유는 R과 M을 GGG 이미지 2장으로 따로 넣고, 출력을 $\mathrm{mean}_c$로 읽었습니다.
- 분할은 $[0,r,m]$ 한 장으로 넣고, 출력의 ch1과 ch2를 그대로 씁니다: $\hat R=\hat y_{rm}^{(1)}$, $\hat M=\hat y_{rm}^{(2)}$.
- 3채널을 유지하므로 G의 구조(첫 conv와 마지막 conv 포함)는 원본과 같습니다.

**③ height는 생성하지 않고, 렌더에는 GT를 씁니다.**

$$
\hat X^{\text{split}}=\big(\hat y_a,\ \hat y_n,\ \hat y_{rm}^{(1)},\ \hat y_{rm}^{(2)},\ H_{\text{GT}}\big)
$$

- 그룹이 albedo / normal / RM이므로 height를 맡는 네트워크가 없습니다.
- 공정하게 비교하려고 **평가에서는 모든 arm(공유 G 포함)의 height를 GT로 맞춘 뒤** 렌더합니다.
- 학습할 때는 공유 G가 height에도 용량을 쓰고, SR height로 렌더합니다. 이 점은 분할 쪽에 조금 유리합니다(§C-3).

**④ 손실과 업데이트는 그룹마다 따로 합니다.** render 항 $\mathcal L_{\text{ren}}(\hat X^{\text{split}},X)$는 **한 번만** 계산합니다.

$$
\mathcal L_g=\frac1B\sum_{b=1}^{B}\Big[w_{\text{pix}}\lVert\hat y_{b,g}-y_{b,g}\rVert_1+w_{\text{feat}}\lVert\phi(\hat y_{b,g})-\phi(y_{b,g})\rVert_1\Big]+w_{\text{gan}}\,\mathcal L_{\text{RaGAN}}(D_g)+w_g\,\mathcal L_{\text{ren}}(\hat X^{\text{split}},X)
$$

$$
\theta_g\leftarrow\mathrm{Adam}_g\big(\nabla_{\theta_g}\mathcal L_g\big),\qquad g\in\{a,n,rm\}
$$

- $\mathcal L_{\text{ren}}$은 세 네트워크 출력 모두에 의존합니다. 그래서 역전파하면 $\partial\mathcal L_{\text{ren}}/\partial\hat y_g$가 각 네트워크로 흘러갑니다. 이것이 **공동 학습**입니다.
- $\theta_g$는 **자기 손실 $\mathcal L_g$의 gradient만** 받습니다. 그래서 render 가중치를 그룹마다 $w_g$로 다르게 줄 수 있습니다.
- $y_{rm}=[0,R,M]$입니다. ch0은 상수 0이라서 pixel 손실에서는 ch0 출력에만 gradient가 가고, R과 M에는 영향을 주지 않습니다.

**⑤ render 가중치는 $w_g=\kappa_g\,w$입니다.**

$$
\kappa_a=\kappa_n=5,\quad\kappa_{rm}=\tfrac53\qquad\Rightarrow\qquad w_a=w_n=50,\quad w_{rm}\approx16.67
$$

이 값의 의미는 하나의 목적함수로 다시 쓰면 분명해집니다. 네트워크마다 Adam이 따로 있고, Adam은 손실 전체에 상수를 곱해도 업데이트가 ($\epsilon$을 빼면) 같습니다. 따라서 $\mathcal L_g$를 $\kappa_g$로 나눠도 결과가 같고, 분할 학습은 다음 **단일 목적함수**를 최소화하는 것과 같습니다.

$$
\mathcal L^{\text{split}}\;\equiv\;\sum_g\frac{1}{\kappa_g}\Big[\text{pix}_g+\text{feat}_g+\text{gan}_g\Big]+w\,\mathcal L_{\text{ren}}
$$

- **albedo, normal:** $\frac{1}{\kappa_a}\cdot\frac1B=\frac1{5B}$로, 공유 run의 평균 분모와 **정확히 같습니다.** 공유 배치에는 이미지가 $5B$장 있고, 분할 배치에는 $B$장 있기 때문입니다.
- **RM:** 공유 쪽 R 이미지는 3채널 평균으로 읽히므로 채널당 render gradient가 $\tfrac13\,\partial\mathcal L/\partial R$입니다. 분할 쪽은 채널 하나라서 $\partial\mathcal L/\partial R$ 그대로 들어갑니다. 그래서 pixel 쪽 5배와 render 쪽 3배가 상쇄되어 $\kappa_{rm}=5/3$입니다.
- 정리하면, 등가 스케일링은 분할 학습이 "공유 run과 같은 식에서 **파라미터만 나눈 것**"이 되도록 맞추는 장치입니다.
- $\kappa_g$는 배치 구성에서만 나오므로 nb와 $B$에 관계없이 같습니다.
- **한계:** 정규화 방식(평균의 분모)만 맞춥니다. VGG와 GAN gradient의 실제 크기는 그룹마다 다릅니다. comp1 gate처럼 gradient norm으로 실측한 보정은 아닙니다.

**⑥ D가 3개로 나뉩니다.**

$$
\mathcal L_{D_g}=\mathcal L_{\text{RaGAN}}^{D}(D_g;\,y_g,\hat y_g)
$$

- 공유 D는 맵이 섞인 $5B=10$장을 봤습니다. 분할 $D_g$는 자기 그룹의 $B=2$장만 봅니다.
- BatchNorm과 RaGAN의 $\mathbb E[D(\cdot)]$를 2장으로 계산하게 되므로 GAN 신호의 분산이 커집니다.
- D는 그룹마다 VGG128(약 14.5M)입니다. nb=23에서는 D:G 비율이 공유와 같은 0.87입니다.

**⑦ (추가 arm `_rmgrey`) RM의 pixel / VGG / D 손실을 회색 이미지로 계산합니다.**

결과를 보고 추가한 arm입니다(`deeptex_split_nb23_rmgrey`, `deeptex_split_nb7_rmgrey`, `rm_loss: grey`). 기본 분할(②)에서는 RM 네트워크의 자기 손실이 묶음 이미지 $[0,R,M]$ 한 장을 봅니다. pixel L1은 채널마다 따로 계산되므로 크기만 다를 뿐 공유와 같지만, **VGG와 D는 세 채널을 섞어 보는 비선형 함수**입니다. 그래서 R이 받는 feature/GAN gradient가 같은 위치의 M과 상수 0 채널에 따라 달라지고, 공유 run(회색 R 이미지와 회색 M 이미지를 따로 봄)과 목적함수가 달라집니다. own 손실에서는 $w_{\text{feat}}=1$이 $w_{\text{pix}}=0.01$보다 100배 크므로 이 차이가 지배적입니다.

$$
\text{own}_{rm}^{\text{grey}}=\text{pix}+\text{feat}+\text{gan}\ \text{ on }\ \big[\,\mathrm{GGG}(R)\,;\ \mathrm{GGG}(M)\,\big]\ (2B\text{장}),\qquad \mathrm{GGG}(R)=\mathrm{tile}(\hat y_{rm}^{(1)},3)
$$

- 네트워크 구조, 입력 $[0,R,M]$, render 항은 그대로입니다. 출력 ch0은 어떤 손실에도 쓰이지 않습니다.
- $D_{rm}$은 회색 R, M 이미지 $2B$장씩을 봅니다(공유 D가 보던 것과 같은 종류).
- **$\kappa_{rm}=5/2$, $w_{rm}=25$:** 출력 변수 하나($R$의 채널 1)로 따지면, own 항은 $2B$장 평균에서 GGG 세 채널을 모두 거쳐 $\frac{1}{2B}$의 가중치를 받고, 공유 run의 채널 하나는 $\frac{1}{15B}$입니다(비 7.5). render는 ⑤와 같이 3배입니다. $\kappa=7.5/3=5/2$. 같은 방식으로 ②의 묶음 이미지는 $\frac{1/(3B)}{1/(15B)}/3=5/3$입니다.
- 이 arm에서 roughness가 회복되면, 기본 분할의 RM 실패는 분할 자체가 아니라 묶음 이미지가 만든 다른 목적함수 때문입니다.

**바뀌지 않는 것:** 데이터 스트림($B=2$, seed 1, shuffle off, crop과 flip/rot 순서), 40k step, $w_{\text{pix}}, w_{\text{feat}}, w_{\text{gan}}$, Adam과 LR 스케줄, 렌더러와 빛 100개, normal 증강(성분은 바꾸지 않음).

---

## B. MatUp

### B0. 기본 논문 (MatUp, EGSR 2024)

재질 하나에 대해 처음부터 최적화합니다(pretrained 없음).

1. LR 맵을 bilinear로 4배 키워 7채널 초기값을 만듭니다.
$$
S_0=\mathrm{bilinear}_{\uparrow4}(a,\,m,\,n_{xy},\,\alpha),\qquad\alpha=r^2
$$
2. LR 렌더를 SwinIR-L로 키워 목표 렌더를 만듭니다: $\tilde I_l=\mathrm{SwinIR}(I^{LR}_l)$.
3. MLP 하나가 17×17 패치를 보고 중심 texel의 잔차를 출력합니다.
$$
\hat S(p)=S_0(p)+f_\varphi\big(P_{17}(S_0;p)\big),\qquad f_\varphi:\mathbb R^{7\times17\times17}\to\mathbb R^7
$$
4. 손실은 render 손실과 metallic 정규화입니다.
$$
\mathcal L(\varphi)=\frac{1}{|\mathcal B|L}\sum_{p\in\mathcal B}\sum_{l}\big\lVert\rho(\hat S(p);\omega_l)-\tilde I_l(p)\big\rVert_2+\alpha_{\text{reg}}\,\mathrm{MSE}\big(\hat m(p),\,m_0(p)\big)
$$

코드에서 확인한 구조(`external/matup_egsr`):

| 항목 | 값 | 위치 |
|---|---|---|
| 입력/출력 채널 | 7 / 7 (albedo 0–2, metallic 3, normal xy 4–5, α 6) | `main_optim.py:59-61`, `data_utils.py:94-101` |
| 이웃 크기 $N$ | 17×17 (`tile_width`) | `main.py:26` |
| 층 | conv 17×17 (7→128), 1×1 (128→128) ×3, 1×1 (128→7), LeakyReLU | `model.py:18-30` |
| 파라미터 | 309,511 | |
| 잔차 | 중심 texel 초기값에 더함 | `train.py:41-46` |
| 정답 | 중심 texel의 SwinIR 렌더, 빛 100개 | `dataLoader.py:180-182` |

### B1. comp3 분할 MLP

MLP를 그룹별로 셋으로 나누고, 각 MLP는 **자기 채널의 패치만** 봅니다.

$$
\Delta_a=f_{\varphi_a}\big(P_{17}(S_0^{\,a})\big)\in\mathbb R^3,\qquad
\Delta_n=f_{\varphi_n}\big(P_{17}(S_0^{\,n_{xy}})\big)\in\mathbb R^2,\qquad
\Delta_{rm}=f_{\varphi_{rm}}\big(P_{17}(S_0^{\,m,\alpha})\big)\in\mathbb R^2
$$

$$
\hat S(p)=S_0(p)+\big[\Delta_a,\ \Delta_{rm}^{(m)},\ \Delta_n,\ \Delta_{rm}^{(\alpha)}\big]\qquad(\text{채널 순서 }a,m,n_{xy},\alpha)
$$

**달라지는 점**
1. **입력 정보:** 공유 MLP의 albedo 출력은 normal과 roughness 패치도 봅니다. 분할 MLP는 자기 맵만 봅니다. 이것이 실험의 핵심 차이입니다.
2. **파라미터:** albedo 161,027개, normal 123,906개, RM 123,906개로 합계 408,839개입니다. 공유의 약 **1.32배**입니다.
   - 파라미터 대부분이 첫 층(입력 채널 × 289 × 128)에 있는데, 입력 채널을 그룹끼리 나눠 갖기 때문에 3배가 되지 않습니다.
   - 각 MLP의 폭(128)과 층 수(4)는 공유 MLP와 같습니다.
3. **(추가 arm) 파라미터를 맞춘 두 변형:** 세 MLP의 폭 $h$만 바꾸고(층 수 4, 타일 17 그대로) 합계를 공유의 1배, 3배에 맞춥니다. 합계는 $9h^2+2042h+7$입니다.
   - `matup_split_x1`: $h=104$, 309,719개 (공유의 **1.001배**). DeepTex nb7(0.93배)에 해당합니다.
   - `matup_split_x3`: $h=227$, 927,302개 (공유의 **2.996배**). DeepTex nb23(3배)에 해당합니다.
   - 기존 `matup_split`($h=128$, 1.32배)은 "그룹마다 공유와 같은 모양의 MLP"라는 점에서 nb23과 같은 설계지만, 첫 층 때문에 파라미터는 1.32배입니다.
4. **(진단 arm) metallic 정규화 끄기:** `matup_shared_noreg`, `matup_split_noreg`는 각각 공유, 분할($h=128$)과 같고 $\alpha_{\text{reg}}=0$만 다릅니다(MatUp `--alpha_reg`, 기본 0.1).
   - 정규화항은 $\alpha_{\text{reg}}\,\overline{(\Pi(z_m)-s_m)^2}$입니다. clamp $\Pi$ 뒤에 걸려 있어 metallic이 경계에 붙으면 gradient가 함께 0이 됩니다.
   - ep10 결과에서 분할 arm의 일부 seed는 채널 하나 이상이 clamp 경계에 붙은 채로 끝났습니다(plastic albedo 전체 0, roofing B 0, brick_wall roughness 1). 이 arm은 그 원인이 정규화항인지 가립니다.
   - 분할에서 albedo MLP는 $z_m$에 닿지 않으므로 $\nabla_{\varphi_a}$(정규화항)$\equiv0$입니다. 그래서 정규화항이 원인이라면 RM 쪽 사망(roughness 고정)만 바뀌고 albedo 사망은 그대로여야 합니다.
   - 짝 비교는 `matup_split_noreg` − `matup_shared_noreg`입니다(정규화 조건이 같은 쌍). 정규화 효과는 같은 seed의 `*_noreg` − 기존 arm으로 봅니다.
5. **(진단 arm) clamp 사망 수정안:** `*_zinit`(마지막 층 0 초기화), `*_st`(학습 중 straight-through clamp)를 공유와 분할에 똑같이 적용합니다. 원인 가설, 수식, 예측, 계측(probe)은 [CLAMP_DEATH.md](CLAMP_DEATH.md)에 있습니다.

**바뀌지 않는 것**
- 손실 $\mathcal L$이 그대로입니다. render 손실은 하나뿐이고 그룹별 항이 없으므로, DeepTex와 달리 $w_g$를 조정할 필요가 없습니다.
- 세 MLP의 파라미터 전체를 Adam 하나로 함께 최적화합니다. MatUp은 원래 공동 학습입니다. Adam은 파라미터마다 따로 적응하므로, Adam을 셋으로 나눈 것과 같습니다.
- 그대로인 것: SwinIR-L(릴리스 가중치), 목표 렌더 $\tilde I_l$(모든 arm과 seed가 같은 캐시를 씀), 빛, epoch 10, 배치 128, 학습률, $\alpha_{\text{reg}}$(B1-4의 `*_noreg`만 0), 초기화와 clamp(B1-5의 `*_zinit`, `*_st`만 다름), height 처리(위치에만 사용).
- 실험상 차이: seed를 3개 돌려 노이즈 수준을 함께 보고합니다.

**SwinIR-L은 왜 나누지 않는가:** MatUp의 SwinIR-L은 맵이 아니라 렌더 이미지를 업스케일합니다. 렌더에는 모든 맵이 섞여 있으므로, 맵 그룹으로 나눌 대상이 아닙니다.

---

## C. 공유와 분할 비교에서 남는 차이

1. **오차 상쇄:** 공동 학습이므로 분할도 공유처럼 렌더 오차를 맵끼리 상쇄할 수 있습니다(예: albedo를 어둡게 해서 normal 오차를 보상). 1차 근사로 쓰면 다음과 같습니다.
$$
\rho(\hat X)-\rho(X)\approx\sum_g J_g\,\delta_g,\qquad J_g=\partial\rho/\partial Y_g
$$
   - 평가에서는 그룹마다 "자기 맵만 SR, 나머지는 GT"로 렌더한 오차 $\mathcal L_{\text{ren}}(\tilde X_g)$도 기록합니다. 상쇄가 있으면 $\mathcal L_{\text{ren}}(\hat X)<\sum_g\mathcal L_{\text{ren}}(\tilde X_g)$가 됩니다.
2. **D의 배치 통계:** §A2-⑥과 같습니다.
3. **height 용량:** 공유 G만 height를 학습합니다. 평가는 모든 arm에서 GT height로 맞춥니다.
4. **MatUp 입력 정보:** §B1-1과 같습니다. DeepTex 분할도 albedo 네트워크는 albedo LR만 봅니다. 그래서 두 실험은 "다른 맵의 정보를 보지 않는 전용 네트워크가 나은가"라는 같은 질문을 던집니다.

## D. 결과 해석

| 결과 | 해석 |
|---|---|
| nb=23 분할 > 공유 | 전문화가 이득입니다. 다만 파라미터가 3배입니다. |
| nb=7 분할 > 공유 | 파라미터가 더 적은데도 이겼으므로, 전문화 효과의 강한 증거입니다. |
| nb=23 분할 > 공유 > nb=7 분할 | 분할은 용량을 더 줄 때만 이득입니다. |
| 분할 ≤ 공유 (둘 다) | 맵 사이 정보 공유가 이득입니다. MatUp 결과와 일치하는지 확인합니다. |

비교 방법: 평가 재질 12개에서 재질별로 짝지은 차이를 봅니다. MatUp은 seed 3개의 평균과 표준편차를 함께 봅니다.
