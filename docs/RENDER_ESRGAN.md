# 렌더 공간 ESRGAN: 공유 vs 분할 (설계)

- 작성: 2026-11-26. **상태: 설계만 있음, 구현 전.**
- 관련 문서: [SPLIT_THEORY.md](SPLIT_THEORY.md) (기존 맵 공간 분할), [DEVIATIONS.md](DEVIATIONS.md)

**왜 새 설계가 필요한가.** 기존 DeepTex 분할(SPLIT_THEORY A2)은 ESRGAN 손실(pixel, VGG, GAN)을 **맵**에 걸고, 렌더 손실을 그 옆에 더했습니다. RM 그룹의 VGG와 D가 `[0, R, M]` 묶음 이미지를 보게 되면서, 분할 쪽 목적함수가 공유 쪽과 달라졌습니다(VGG 첫 층과 D가 채널을 섞음). 그래서 "네트워크 분할의 효과"와 "목적함수가 달라진 효과"를 구분할 수 없었습니다.

이 설계는 ESRGAN 손실을 **렌더 이미지**에 겁니다. 손실 전체가 합쳐진 재질의 렌더만의 함수가 되므로, 공유와 분할의 목적함수가 정확히 같아집니다. MatUp 분할처럼 "네트워크만 다른" 비교가 됩니다.

---

## 1. 이론

### 1.0 기호

| 기호 | 뜻 |
|---|---|
| $x=(x_a,x_n,x_R,x_M)$ | LR 맵 (평가 128², 학습 crop 32²). $x_R, x_M$은 GGG 회색 이미지 |
| $y$, $\hat y$ | GT HR 맵, SR 맵 (4맵. height는 예측하지 않음) |
| $U(\cdot)$ | bilinear ×4 업샘플 (MatUp의 $S_0$와 같은 연산, cv2 INTER_LINEAR) |
| $\mathcal R_l(\cdot)$ | 조명 $l$에서의 미분 가능한 렌더 (MatUp GGX의 TF 포트, GT height 사용) |
| $\tau$ | 표시 변환 (Reinhard + gamma). $I_l=\tau(\mathcal R_l(y))$, $\hat I_l=\tau(\mathcal R_l(\hat y))$ |
| $\mathcal L_{\text{all}}$ | MatUp 학습 조명 100개. $S\subset\mathcal L_{\text{all}}$은 step마다 고르는 $K$개 |
| $\Omega$ | GT normal이 유효한 텍셀 |
| $\varphi$, $C$ | VGG19 특징(VGG54, 활성화 전), D의 logit |

### 1.1 생성기와 구조적 앵커 (A: bilinear 잔차)

$$
\text{공유: }\hat y_k=U(x_k)+G_\theta(x_k),\ k\in\{a,n,R,M\}\qquad
\text{분할: }\hat y_g=U(x_g)+G_{\theta_g}(x_g),\ g\in\{a,n,rm\}
$$

**① 모든 arm이 같은 점에서 출발합니다.** 마지막 conv를 $W_L=0$, $b_L=0$으로 초기화합니다.

$$
G(x)=0\quad\Rightarrow\quad \hat y^{(0)}=U(x)\qquad(\text{공유, 분할 nb23, nb7 모두 같음})
$$

MatUp이 $S_0$에서 출발하는 것과 같습니다. 출발 시점의 출력이 arm 사이에서 정확히 같으므로, 이후의 차이는 모두 학습에서 생긴 것입니다.

**② 0 초기화여도 학습은 시작됩니다.**

$$
\frac{\partial\hat y}{\partial W_L}=h_{L-1}(x)\neq0,\qquad \frac{\partial\hat y}{\partial\theta_{<L}}=W_L^{\top}(\cdots)=0
$$

첫 step에는 마지막 층만 움직이고, $W_L\neq0$이 되는 순간부터 gradient가 전체로 흐릅니다.

**③ 앵커의 의미.** G가 배우는 것은 잔차 $\Delta=\hat y-U(x)$입니다.

- 색, 밝기, metallic 0 같은 입력 수준이 기본값이 됩니다. 이를 바꾸려면 G가 0이 아닌 저주파 $\Delta$를 일부러 만들어야 합니다.
- 잔차가 없으면 G가 항등 사상(색 포함)부터 스스로 만들어야 합니다. 학습 분포 밖의 색에서 외삽 오류가 생깁니다(기존 결과의 foam 색조 이동).
- 축소가 4×4 box 평균(INTER_AREA)이므로 $U(x)$는 이미 거의 LR과 일관됩니다: $\mathrm{box}_4(U(x))\approx x$.
- 파라미터 수는 늘지 않습니다.

### 1.2 손실

$$
\mathcal L_G(\hat y)=\lambda_r\,\underbrace{\frac{1}{|\Omega|\,|\mathcal L_{\text{all}}|}\sum_{p\in\Omega}\sum_{l\in\mathcal L_{\text{all}}}\big\|\hat I_l(p)-I_l(p)\big\|_2}_{\ell_{2,1}\ (\text{ESRGAN pixel 항의 자리})}
+\lambda_f\,\underbrace{\frac1K\sum_{l\in S}\big\|\varphi(\hat I_l)-\varphi(I_l)\big\|_1}_{\text{VGG: 렌더끼리}}
+\lambda_g\,\mathcal L^{G}_{\text{Ra}}
\;\big[+\,w_m\,\|\hat m-U(x)_m\|_2^2\big]
$$

$$
\mathcal L^{D}_{\text{Ra}}=\mathrm{BCE}\big(1,\;C(I_S)-\overline{C(\hat I_S)}\big)+\mathrm{BCE}\big(0,\;C(\hat I_S)-\overline{C(I_S)}\big)
$$

$$
\mathcal L^{G}_{\text{Ra}}=\mathrm{BCE}\big(0,\;C(I_S)-\overline{C(\hat I_S)}\big)+\mathrm{BCE}\big(1,\;C(\hat I_S)-\overline{C(I_S)}\big)
$$

- 가로줄은 B·K장 배치 평균이고, BCE는 logit 입력(`sigmoid_cross_entropy_with_logits`)입니다.
- **맵 공간 손실은 없습니다.**
- 대괄호 항은 MatUp의 metallic 정규화($\alpha_{\text{reg}}=0.1$)입니다. 설정값 $w_m$으로 두고 기본은 0입니다. `render_tf.py`는 "생성기에는 텍셀별 출발점이 없어서" 이 항을 옮기지 않았는데, 앵커 A가 그 출발점 $U(x)$를 만들어 주므로 이제 그대로 옮길 수 있습니다.

**④ 조명 부분집합은 확률적 추정량입니다.** VGG와 GAN 항은 모든 조명에 대한 평균을 목표로 합니다. $S$를 균등하게 뽑으면 불편 추정량입니다.

$$
\mathbb E_S\Big[\frac1K\sum_{l\in S}f_l\Big]=\frac{1}{100}\sum_{l\in\mathcal L_{\text{all}}}f_l
$$

grazing과 정면 조명을 강제로 넣는 층화 추출은 조명별 가중치 $\pi_l$을 바꿉니다. 렌더에 잘 안 드러나는 normal, roughness 신호를 키우기 위한 의도한 가중이고, 모든 arm이 같은 $\pi$를 씁니다.

### 1.3 목적함수 동일성 (이 실험의 핵심)

**정리.** $\mathcal L_G$는 파라미터에 SR 맵 $\hat y$를 통해서만 의존합니다.

$$
\mathcal L_G=F(\hat y;\,D),\qquad
\text{공유: }F\big(\Phi_{\text{sh}}(\theta)\big),\qquad
\text{분할: }F\big(\Phi_{\text{sp}}(\theta_a,\theta_n,\theta_{rm})\big)
$$

두 arm은 같은 $F$를 서로 다른 함수 집합 $\Phi$ 위에서 최소화합니다. 남는 차이는 표현력(구조와 파라미터 수)과 그에 따른 최적화 경로뿐입니다.

**⑤ κ와 그룹별 가중치가 필요 없습니다.** 손실이 하나라 그룹마다 맞출 정규화가 없습니다. Adam은 원소별 연산이라, 파라미터를 그룹으로 나눠 Adam을 따로 두어도 하이퍼파라미터와 step 수가 같으면 하나의 Adam과 업데이트가 같습니다.

$$
\theta_i\leftarrow\theta_i-\eta\,\frac{\hat m_i}{\sqrt{\hat v_i}+\epsilon},\qquad \hat m_i,\hat v_i\text{는 }\partial F/\partial\theta_i\text{만의 함수}
$$

**⑥ 기존 설계와의 차이.** 기존 분할의 RM 자기 손실은 $\varphi([0,\hat R,\hat M])$, $D_{rm}([0,\hat R,\hat M])$로, 공유의 $\varphi(\mathrm{GGG}(\hat R))$, $\varphi(\mathrm{GGG}(\hat M))$과 다른 함수였습니다. 이번에는 VGG와 D의 입력이 $\hat y$로부터 정해지는 렌더 $\hat I$뿐입니다.

**⑦ D도 같은 게임입니다.** D는 두 arm 모두 같은 종류의 입력(렌더 B·K장)을 받고, RaGAN 평균도 같은 크기의 배치에서 계산됩니다.

### 1.4 gradient의 분배: 렌더러가 정합니다

$$
\frac{\partial\mathcal L_G}{\partial\theta_g}=\sum_{l}\underbrace{\frac{\partial\mathcal L_G}{\partial\hat I_l}}_{\text{l2,1 / VGG / D의 보정}}\ \underbrace{\frac{\partial\hat I_l}{\partial\hat y_g}}_{\text{렌더러 야코비안}}\ \underbrace{\frac{\partial\hat y_g}{\partial\theta_g}}_{\text{G}}
$$

$$
\mathcal R_l\approx\Big[(1-m)\frac{a}{\pi}+\frac{D_\alpha(n\!\cdot\!h)\,F(F_0)\,G_\alpha}{4(n\!\cdot\!l)(n\!\cdot\!v)}\Big](n\!\cdot\!l)E_l,\qquad \alpha=R^2,\ F_0=\mathrm{lerp}(0.04,a,m)
$$

| 맵 | 렌더러 야코비안 | 신호가 큰 곳 |
|---|---|---|
| $a$ | diffuse에 거의 선형, 금속이면 $F_0$에도 | 거의 모든 텍셀과 조명 |
| $n$ | $n\!\cdot\!l$, $n\!\cdot\!h$ | grazing 조명, 굴곡 |
| $R$ | $D_\alpha$, $G_\alpha$ | 하이라이트, 금속 |
| $M$ | diffuse/specular 비율, specular 색 | 밝은 albedo + specular |

**⑧ 관측 불가능한 성분과 앵커의 역할 분담.** $\|\partial\hat I/\partial\hat y_k\|\approx0$인 성분(비금속의 roughness, 어두운 albedo 위의 metallic)에는 손실 신호가 없습니다.

- **렌더 손실:** 렌더에 보이는 성분을 GT 쪽으로 당깁니다.
- **앵커 A:** 렌더에 안 보이는 성분이 bilinear에서 멀어지지 않게 합니다. 신호가 없으면 $\Delta$를 만들 이유가 없다는 암묵적 효과입니다.

MatUp도 같은 구조입니다. 기존 결과의 roughness 표류(분할 −6 dB)는 이 기본값이 없어서 생겼다는 가설과 맞습니다.

**⑨ clamp.** 렌더러가 $a, m\in[0,1]$, $R\ge0.045$로 자르므로 범위 밖에서는 야코비안이 0입니다. 출력이 범위 안의 $U(x)$에서 시작하므로 갇힐 위험은 기존보다 작습니다. `oor_*` 비율로 감시합니다.

### 1.5 배치와 분산

$$
\nabla_{\theta_g}\mathcal L_G=\frac{1}{B}\sum_{i=1}^{B}\Big[\lambda_r\nabla\ell_{2,1}^{(i)}+\frac{\lambda_f}{K}\sum_{l\in S}\nabla\text{VGG}_l^{(i)}+\cdots\Big]
$$

- 분할 $G_g$도 공유와 **같은 B개 재질, 같은 렌더**(조명 100개 + K개)에서 gradient를 받습니다. 기존 설계에서 네트워크당 배치가 10장에서 2장으로 줄던 문제가 사라집니다.
- RaGAN 평균은 B·K장(기본 16장)에서 계산됩니다. 기존 $D_{rm}$의 2장보다 훨씬 안정적입니다.

### 1.6 λ 보정의 근거

Adam은 손실 전체의 상수배에 불변이므로 $\lambda_r=1$로 고정해도 되고, 비율 $\lambda_f/\lambda_r$, $\lambda_g/\lambda_r$만 의미가 있습니다. comp2 레시피에서 "렌더 대비 지각/적대 신호의 상대 세기"를 렌더 공간에서 재현하도록 정합니다.

$$
\rho_f^{\text{map}}=\frac{\|\nabla_{\hat y}\,\lambda_f^{\text{map}}\,\text{VGG}_{\text{map}}\|}{\|\nabla_{\hat y}\,10\,\ell_{2,1}\|}\quad(\text{renloss\_21 설정, 처음부터 300 step})
$$

$$
\lambda_f=\rho_f^{\text{map}}\cdot\frac{\|\nabla_{\hat y}\,\ell_{2,1}\|}{\|\nabla_{\hat y}\,\text{VGG}_{\text{ren}}\|}\quad(\text{ren\_shared, }\lambda=1\text{로 300 step}),\qquad \lambda_g\text{도 같은 방식}
$$

gradient 크기를 SR 맵 $\hat y$에서 재므로 공유와 분할 어느 쪽에도 유리하지 않습니다. 모든 arm이 같은 λ를 씁니다.

### 1.7 MatUp과의 대응

| | MatUp | 렌더 공간 ESRGAN |
|---|---|---|
| 출발점 | $S_0=U(x)$ | $U(x)$ (0 초기화) |
| 학습 대상 | MLP 잔차 $\Delta$ | RRDB 잔차 $G(x)$ |
| 렌더 손실 | $\ell_{2,1}$ (100개) | $\ell_{2,1}$ (100개) |
| 목표 렌더 | SwinIR이 올린 LR 렌더 | GT 렌더 |
| 지각/적대 항 | 없음 | 렌더 VGG + 렌더 RaGAN (K개) |
| metallic 정규화 | $0.1\cdot\mathrm{MSE}(\hat m,m_0)$ | 선택 (기본 0) |
| 학습 범위 | 재질 하나씩 최적화 | 학습셋 전체 → 일반화 |
| 분할 | MLP를 채널 그룹별로 | RRDB를 맵 그룹별로 |
| 목적함수 동일성 | 성립 | 성립 (1.3) |

### 1.8 예측 (검증할 가설)

1. **두 arm 모두 roughness 표류가 사라집니다** (앵커 A, ⑧). roughness PSNR이 bilinear 수준 이상이어야 합니다.
2. **분할 효과가 순수하게 측정됩니다.** 원래 가설이 맞다면 only-g 지표와 맵 PSNR에서 분할 > 공유가 나와야 합니다.
3. **nb23 대 nb7:** 목적함수가 같으므로 파라미터 수의 효과만 남습니다. 기존 결과(nb7 > nb23)가 GAN 불안정 때문이었다면 이번에는 뒤집혀야 합니다.

### 1.9 한계

- **분해 모호성:** 렌더가 같은 맵 조합은 손실로 구분되지 않습니다. 음영이 albedo에 구워질 수 있으며, only_a 지표와 몽타주로 감시합니다.
- **관측 불가능한 성분은 bilinear 수준에 머뭅니다.** 비금속 roughness의 HR 디테일은 복원되지 않습니다. 렌더 기준으로는 최적이지만 맵 PSNR 기준으로는 한계입니다.
- **조명 가중:** 층화 추출이 grazing과 정면 조명에 가중을 줍니다(④).
- **고정된 것:** GT height, 학습 조명 100개 분포, tonemap의 하이라이트 압축.
- **D의 환각:** 렌더 GAN이 존재하지 않는 굴곡을 지어낼 수 있습니다. held-out 조명 지표와 normal MAE로 감시합니다.

---

## 2. 구현 계획

### 2.0 확정된 설계

| 항목 | 결정 |
|---|---|
| 구조적 앵커 | A: $\hat y=U(x)+G(x)$, 마지막 conv 0 초기화 |
| 손실 | $\lambda_r\,\ell_{2,1}$(100개) + $\lambda_f\,\mathrm{VGG}_{\text{render}}$ + $\lambda_g\,\mathrm{RaGAN}_{\text{render}}$ ($K=8$). 맵 공간 손실 없음 |
| D | 1개, 렌더만 판정 (B·K장) |
| arm | `ren_shared` (nb23 1개, 4맵, height 없음), `ren_split_nb23` (3배), `ren_split_nb7` (0.93배) |
| 설정값으로 열어 둠 | $w_m$ (기본 0, MatUp과 같게 하려면 0.1), B (기본 2), K (기본 8), λ (보정 게이트에서 정함) |

### 2.1 새 코드: `variants/deeptex/render_esrgan/`

| 파일 | 내용 |
|---|---|
| `residual.py` | `ResidualG(rrdb, scale)`: Keras 모델로 감싸 $U(x)+\mathrm{rrdb}(x)$. 체크포인트와 추론이 같은 래퍼를 씀. `zero_last_conv(rrdb)`. bilinear는 `tf.image.resize`(half-pixel) |
| `lights.py` | `light_subset(step, seed, K)`: 고도 기준 grazing 2개 이상, 정면 쪽 2개 이상, 나머지 무작위. (seed, step)으로 결정되어 재개해도 같음 |
| `render_images.py` | `render_images(maps, pos, lights_S)`: `render_tf.render_display`로 `(N,3,K)`를 렌더해 `(B·K,128,128,3)`로 변환. GT normal 무효 텍셀은 가짜 렌더에서도 GT 값으로 덮음 |
| `losses.py` | RaGAN G/D (logit 기반, 포화 안전). VGG는 upstream `ContentLoss` 재사용(입력 0~1). `metal_reg` ($w_m=0$이면 꺼짐) |
| `train.py` | 학습기 하나로 `arch: shared` / `split` 모두 처리 (2.2) |
| `calibrate.py` | λ 보정 게이트 (2.4) |
| `test_render_esrgan.py` | CPU 단위 테스트 (2.5) |
| `ren_shared.yaml`, `ren_split_nb23.yaml`, `ren_split_nb7.yaml` | λ는 `null`. 보정 전에는 학습기가 시작을 거부함 (comp2 방식) |

### 2.2 학습기 한 step

```python
gt_all, valid = render_loss.target(hr5, pos)                 # 조명 100개, tape 밖
S    = light_subset(step, seed, K)
real = render_images(decode(hr5), pos, lights[S])            # (B*K,128,128,3), tape 밖
with tf.GradientTape(persistent=True) as tape:
    if arch == "shared":   # G 하나가 4맵을 각각 이미지로 처리 (배치 4B)
        sr4 = G(concat[x_a, x_n, x_R, x_M]);  sr5 = stack(split(sr4) + [H_GT])
    else:                  # G_a, G_n, G_rm
        sr5 = grp.compose({g: G[g](x[g]) for g in GROUPS}, hr5)
    l21  = render_loss.loss(sr5, pos, gt_all, valid)
    fake = render_images(decode(sr5), pos, lights[S])
    feat = vgg(real, fake)
    gan  = ragan_G(D(real), D(fake))
    L_G  = lr * l21 + lf * feat + lg * gan + wm * metal_reg(sr5, lr5)
    L_D  = ragan_D(D(real), D(tf.stop_gradient(fake)))
grads_G = tape.gradient(L_G, all G vars)    # D -> 렌더러 -> compose -> 각 G
grads_D = tape.gradient(L_D, D vars)
# gradient 전부 유한한지 먼저 확인 -> Adam (G마다 1개, D 1개)
```

데이터 스트림, Adam/LR, 체크포인트, 재개, NaN 처리는 `split/train.py`를 그대로 가져옵니다.

**로그**
- `[terms]`: l21, feat, gan, D, metal_reg
- `[render]`: 공동 l21과 only-g l21 (기존 모니터 재사용)
- `oor_*`: 범위 밖 비율
- `[gnorm]`: 500 step마다 각 항이 SR 맵 슬롯(a/n/R/M)에 주는 gradient 크기

### 2.3 셸 스크립트: `scripts/render_esrgan.sh`

```
test                 CPU 단위 테스트
calibrate            λ 보정 게이트 -> runs/deeptex/render_esrgan/calibration.json (백그라운드)
smoke ARM            200 step (백그라운드)
train ARM            40k step (백그라운드, 재실행하면 이어서)
status / stop STAGE ARM
ARM: shared | nb23 | nb7
```

### 2.4 λ 보정 게이트 (`calibrate.py`)

1. renloss_21 설정으로 처음부터 300 step: $\rho_f^{\text{map}}$, $\rho_g^{\text{map}}$ 측정 (SR 맵에서의 gradient 크기 비율)
2. `ren_shared`를 $\lambda_r=\lambda_f=\lambda_g=1$로 300 step: $\|\nabla\text{VGG}_{\text{ren}}\|$, $\|\nabla\text{gan}_{\text{ren}}\|$, $\|\nabla\ell_{2,1}\|$ 측정
3. 1번 비율이 재현되도록 $\lambda_f, \lambda_g$를 정해 `calibration.json`과 세 yaml에 기록
4. 학습기는 yaml의 λ가 `calibration.json`과 다르면 시작을 거부

### 2.5 단위 테스트 (CPU, 시작 전 필수)

1. **잔차 래퍼:** 0 초기화 직후 출력 == bilinear, 그 bilinear ≈ cv2 INTER_LINEAR, 마지막 conv의 gradient ≠ 0
2. **목적함수 동일성:** 같은 SR 맵을 주면 `shared`와 `split` 경로의 $\mathcal L_G$가 완전히 같음 (이 실험의 핵심 보장)
3. **연쇄법칙:** 작은 G에서 `tape.gradient` == 수치 미분, 각 $G_g$가 렌더 gradient를 받음
4. **조명 부분집합:** 같은 (seed, step)이면 같은 조명, grazing과 정면 조명 포함
5. **렌더 이미지 변환:** shape, 무효 텍셀 처리, 조명 100개로 했을 때 l2,1 경로와 같은 값
6. **RaGAN logit 구현:** 포화되지 않은 구간에서 upstream `GeneratorLoss`/`DiscriminatorLoss`와 같은 값
7. **파라미터 수:** 래퍼가 파라미터를 늘리지 않음

### 2.6 평가 연결

| 파일 | 변경 |
|---|---|
| `variants/eval/arms.py` | kind `ren_shared`, `ren_split`. 비교 쌍 `ren_split_nb23 – ren_shared`, `ren_split_nb7 – ren_shared` |
| `variants/eval/infer_deeptex.py` | `ResidualG`로 모델 생성. `ren_shared`는 4맵만 출력(height 작업 제외). split은 `run_split` 재사용 |
| `variants/eval/montage.py` | 실험 `deeptex_render_esrgan_*` 따로. 행: GT / bilinear / ren_shared / ren_split_nb23 / ren_split_nb7 |
| `scripts/eval.sh` | status에 새 체크포인트 경로 |

추론은 기본적으로 타일 없이 이미지 전체로 합니다(128² 전체에 bilinear, MatUp의 $S_0$와 같음). 학습은 32² crop이라 가장자리 1~2 텍셀의 bilinear가 조금 다릅니다.

### 2.7 문서

- 이 문서를 구현에 맞춰 갱신합니다.
- `COMMANDS.txt` [7]: test → calibrate → smoke → train → 평가
- `README.md` 표에 한 줄 추가

### 2.8 진행 순서와 통과 기준

| 단계 | 어디서 | 통과 기준 |
|---|---|---|
| 0. 대기 중인 변경 커밋 | 이 PC | rmgrey, MatUp 1배/3배를 먼저 별도 커밋 |
| 1. 구현 + 단위 테스트 | 이 PC → 연구실 PC `test` | 2.5 전부 OK |
| 2. λ 보정 | 연구실 PC `calibrate` | `calibration.json` 생성 |
| 3. smoke | 세 arm 각 200 step | EXIT 0, step당 시간과 메모리 기록. OOM이면 K 축소 또는 `light_chunks` |
| 4. 본 학습 | `ren_shared` → `ren_split_nb7` → `ren_split_nb23` | 각 40k, 한 번에 하나 |
| 5. 평가 | infer → montage → score | 새 실험 몽타주와 비교 쌍 표 |
