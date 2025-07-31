### Derivation of the Optimal Work Arrangement {#sec-appendix-g-cd}

This appendix derives the optimal remote work policy, $\alpha^*(h, \psi)$, and analyzes the properties of the threshold functions that partition the labor market into distinct work arrangement regimes.

#### The Maximization Problem

Under the assumption of efficient bargaining, the firm and worker choose the remote work share $\alpha \in$ to maximize the total joint flow value of the match. With quasi-linear utility, this simplifies to maximizing the sum of output $Y(\alpha)$ minus the worker's in-office disutility $c(1-\alpha)$:
$$
\max_{\alpha \in [0,1]} \quad \Big\{Y(\alpha \mid h, \psi) - c(1-\alpha)\Big\}
$$
To solve this problem, we use the functional forms specified in the main text:

*   **Production Function:** $Y(\alpha \mid h, \psi) = A(h) \cdot \left[(1 - \alpha) + \alpha \cdot g(h, \psi)\right]$
*   **Baseline Productivity:** $A(h) = A_1 h$
*   **Relative Remote Productivity:** $g(h, \psi) = \psi_0 h^{\phi} \psi^{\nu}$
*   **In-Office Cost Function:** $c(1-\alpha) = c_0 \frac{(1-\alpha)^{1+\chi}}{1+\chi}$

The First-Order Condition (FOC) is found by differentiating the joint surplus $S(\alpha)$ with respect to $\alpha$ and setting it to zero.

*   **Derivative of Output:** $\frac{dY}{d\alpha} = A_1 h (g(h, \psi) - 1)$
*   **Derivative of Cost:** $\frac{d}{d\alpha} \left( -c(1-\alpha) \right) = - \left( c_0(1-\alpha)^\chi \cdot (-1) \right) = c_0(1-\alpha)^\chi$

The FOC is therefore:
$$
A_1 h (g(h, \psi) - 1) + c_0(1-\alpha)^{\chi} = 0
$$
Rearranging this gives the fundamental equation for an interior solution:
$$
A_1 h (1 - g(h, \psi)) = c_0(1-\alpha)^{\chi}
$$
For an interior solution to exist, the left-hand side must be positive, which requires $1 - g(h, \psi) > 0$, or $g(h, \psi) < 1$. This means remote work must be less productive than in-person work for a hybrid arrangement to be optimal.

#### Optimal Policy and Thresholds

**Interior Solution**

Solving the FOC for the interior solution $\alpha_{\text{interior}} \in (0,1)$:
$$
\alpha_{\text{interior}}(h, \psi) = 1 - \left[ \frac{A_1 h \left( 1 - g(h, \psi) \right)}{c_0} \right]^{\frac{1}{\chi}} = 1 - \left[ \frac{A_1 h \left( 1 - \psi_0 h^\phi \psi^\nu \right)}{c_0} \right]^{\frac{1}{\chi}}
$$

**Threshold Derivations**

The boundaries of the hybrid region are determined by analyzing the derivative of the surplus, $\frac{d(Y-c)}{d\alpha}$, at the corners $\alpha=0$ and $\alpha=1$.

1.  **Upper Threshold, $\overline{\psi}(h)$:** The boundary for full remote work ($\alpha^*=1$) is where 
    
    \begin{align*}
    A_1 h (g(h, \overline{\psi}(h)) - 1) + c_0(1-1)^\chi &= 0 \\
    A_1 h (g(h, \overline{\psi}(h)) - 1) &= 0 \\
    \implies \overline{\psi}(h) &= \psi_0^{-1/\nu} h^{-\phi/\nu}
    \end{align*}
    
2.  **Lower Threshold, $\underline{\psi}(h)$:** The boundary for full in-person work ($\alpha^*=0$) is where 
    
    \begin{align*}
    A_1 h (g(h, \underline{\psi}(h)) - 1) + c_0(1-0)^\chi &= 0 \\
    A_1 h (g(h, \underline{\psi}(h)) - 1) &= -c_0 \\
    g(h, \underline{\psi}(h)) &= 1 - \frac{c_0}{A_1 h} \\
    \psi_0 h^\phi [\underline{\psi}(h)]^\nu &= 1 - \frac{c_0}{A_1 h} \\
    \implies \underline{\psi}(h) &= \left( \frac{1}{\psi_0 h^\phi} \left( 1 - \frac{c_0}{A_1 h} \right) \right)^{1/\nu}\\
    \implies \underline{\psi}(h) & = \overline{\psi}(h)\ \left( 1 - \frac{c_0}{A_1 h} \right)^{1/\nu}
    \end{align*}
    
#### Monotonicity and Properties

**Monotonicity of the Thresholds**

*   **Upper Threshold $\overline{\psi}(h)$:**
    $$
    \frac{\partial \overline{\psi}(h)}{\partial h} = \psi_0^{-1/\nu} \left(-\frac{\phi}{\nu}\right) h^{-\frac{\phi}{\nu}-1}
    $$
    Assuming skill-remote complementarity ($\phi > 0$), the upper threshold is **strictly decreasing** in worker skill. Higher-skilled workers require a lower level of firm efficiency to make full remote work optimal.

*   **Lower Threshold $\underline{\psi}(h)$:** The shape of this corrected threshold is now more complex, let $f(h) = \frac{1}{\psi_0} \left( h^{-\phi} - \frac{c_0}{A_1}h^{-\phi-1} \right)$, so that $\underline{\psi}(h) = [f(h)]^{1/\nu}$. The monotonicity is determined by the sign of $f'(h)$:
    $$
    f'(h) = \frac{1}{\psi_0} \left( -\phi h^{-\phi-1} + \frac{c_0(\phi+1)}{A_1}h^{-\phi-2} \right) = \frac{h^{-\phi-2}}{\psi_0} \left( -\phi h + \frac{c_0(\phi+1)}{A_1} \right)
    $$
    Setting the term in the parenthesis to zero gives the critical point $\hat{h}$:
    $$
    \hat{h} = \frac{c_0}{A_1} \frac{\phi+1}{\phi} = \frac{c_0}{A_1} \left(1 + \frac{1}{\phi}\right)
    $$
    The shape of the threshold now depends critically on the value of the skill-remote elasticity $\phi$. Consider the term:
$$
T(h) = -\phi h + \frac{c_0(\phi+1)}{A_1}
$$
	The sign of $\frac{\partial\underline{\psi}(h)}{\partial h}$ is the same as the sign of $T(h)$. This term captures the tension between two economic forces:

	1.  **The Skill-Remote Interaction Effect ($-\phi h$):** This term reflects how a worker's skill directly alters their productivity in a remote setting. Its effect depends critically on the sign of $\phi$.
	2.  **The Opportunity Cost Effect ($\frac{c_0(\phi+1)}{A_1}$):** This term reflects the cost of forgoing a worker's full baseline productivity ($A_1 h$) by having them work remotely. The firm must be compensated for this loss by the worker's increased effectiveness or willingness to accept a lower wage (captured by $c_0$).

	Before analyzing the cases, let's define our key terms based on the cross-partial derivative of the remote productivity function, $g(h, \psi) = \psi_0 h^\phi \psi^\nu$:
$$
\frac{\partial^2 g}{\partial h \partial \psi} = \psi_0 \phi \nu h^{\phi-1} \psi^{\nu-1}
$$
	The sign of this expression is determined entirely by the sign of $\phi$.
	*   **Complementarity ($\phi > 0$):** Worker skill and firm remote-efficiency are complements. An increase in firm efficiency ($\psi$) raises the marginal productivity of worker skill ($h$) in remote work, and vice-versa.
	*   **Substitutability ($\phi < 0$):** Worker skill and firm remote-efficiency are substitutes. An increase in firm efficiency *lowers* the marginal productivity of worker skill in remote work. This can be thought of as a situation where the firm's technology is so effective that it makes the worker's innate skill less relevant for remote tasks.
- **Case 1: Strong Complementarity ($\phi > 0$):** The lower threshold $\underline{\psi}(h)$ has an **inverted U-shape**, peaking at $\hat{h} = \frac{c_0}{A_1}(1 + \frac{1}{\phi})$.
	*   **Economic Intuition:** This is the most intuitive case. When skill and firm technology are complements, the threshold's shape is driven by a trade-off that changes with the skill level.
	    *   **For Low-Skill Workers ($h < \hat{h}$):** At low skill levels, the complementarity effect ($h^\phi$) is weak and the worker's baseline productivity ($A_1 h$) is low. As skill $h$ increases from a low base, the **opportunity cost effect** dominates. The loss of baseline productivity from not being in the office grows faster than the gain from the weak complementarity. To justify offering even a small amount of remote work, the firm requires progressively higher remote efficiency ($\psi$). Therefore, the threshold $\underline{\psi}(h)$ is **increasing**.
	    *   **For High-Skill Workers ($h > \hat{h}$):** At high skill levels, the **skill-remote interaction effect** becomes dominant. The complementarity is now powerful; a small increase in $h$ makes the worker significantly more effective with the firm's remote technology. This strong gain in remote productivity now outweighs the linear increase in opportunity cost. The firm is willing to offer a hybrid arrangement even with a lower level of its own remote efficiency ($\psi$) because the worker's high skill compensates for it. Therefore, the threshold $\underline{\psi}(h)$ is **decreasing**.
- **Case 2: Weak Substitutability ($-1 < \phi < 0$):** The lower threshold $\underline{\psi}(h)$ is **strictly increasing**.
	 - **Economic Intuition:** In this regime, worker skill and firm technology are substitutes. An increase in worker skill has two negative consequences for the firm's incentive to offer remote work:
	    1.  The **opportunity cost** of not having the worker in the office ($A_1 h$) increases, making in-person work more valuable.
	    2.  The **relative remote productivity** ($h^\phi$) actually *decreases* as $h$ rises, because $\phi$ is negative. The worker becomes comparatively worse at remote work as their skill increases.
    - Both economic forces push in the same direction. As a worker's skill increases, they become simultaneously more valuable in the office and less effective remotely. To overcome this "double penalty" and still find it optimal to offer a hybrid arrangement, the firm must possess a substantially higher level of its own remote efficiency ($\psi$). Consequently, the minimum required efficiency, $\underline{\psi}(h)$, must be **strictly increasing** with worker skill.

**Case 3: Strong Substitutability ($\phi < -1$)** The lower threshold $\underline{\psi}(h)$ is **U-shaped**, with a minimum at $\hat{h} = \frac{c_0}{A_1}(1 + \frac{1}{\phi})$.

*   **Economic Intuition:** This is the most complex case. While skill and technology are still substitutes, the relationship is so strong that it reverses the pattern seen in **Case 2**.
    *   **For Low-Skill Workers ($h < \hat{h}$):** At very low skill levels, the **strong substitutability** dominates. An increase in skill $h$ makes the worker so much relatively worse at remote work (the $h^\phi$ term with $\phi < -1$ falls very rapidly) that this effect outweighs the rising opportunity cost. To compensate for this sharp decline in relative remote fitness, the firm needs less of its own efficiency ($\psi$) to be indifferent, as the trade-off is already heavily skewed toward in-person work. The threshold $\underline{\psi}(h)$ is **decreasing**.
    *   **For High-Skill Workers ($h > \hat{h}$):** As skill becomes sufficiently high, the logic from Case 2 takes over. The **opportunity cost effect** begins to dominate again. The loss of a highly productive worker from the office becomes the primary concern for the firm. Even though the worker is a substitute for technology, their high baseline productivity makes keeping them in the office very attractive. To entice the firm to offer a hybrid arrangement, the firm's own remote efficiency ($\psi$) must be increasingly high. Therefore, the threshold $\underline{\psi}(h)$ is **increasing**.
Of course. This is an excellent next step, as the size of the hybrid region is a key outcome of the model that determines how many worker-firm pairs can even consider a non-corner solution.


#### **Size and Properties of the Hybrid Region**

The existence of a hybrid work arrangement is possible for a worker of skill $h$ only if there is a non-empty range of firm efficiencies $\psi$ such that $\underline{\psi}(h) < \psi < \overline{\psi}(h)$. We define the size of this hybrid region as the width of this interval:
$$
\Delta\psi(h) = \overline{\psi}(h) - \underline{\psi}(h) = \left( \psi_0^{-1/\nu} h^{-\phi/\nu} \right) \left[ 1 - \left( 1 - \frac{c_0}{A_1 h} \right)^{1/\nu} \right]
$$
This region is well-defined for all $h$ such that $1 - \frac{c_0}{A_1 h} > 0$, which requires $h > c_0/A_1$. For skill levels below this minimum, no remote work is ever optimal.

**Analysis of the Hybrid Region's Size**

To understand how the range of hybrid opportunities changes with worker skill, we analyze the derivative $\frac{d(\Delta\psi(h))}{dh}$. The behavior depends critically on the nature of the skill-remote interaction, governed by $\phi$.

- **Case 1: Strong Complementarity ($\phi > 0$):** The size of the hybrid region, $\Delta\psi(h)$, is **strictly decreasing** in worker skill $h$.
	*   **Economic Intuition:** When skill and firm technology are complements, higher-skilled workers are pushed towards the full-remote corner solution more rapidly than they are pulled away from the full-in-person corner.
    1.  **Upper Threshold Effect:** The upper threshold $\overline{\psi}(h)$ is decreasing in $h$. As a worker's skill increases, their high remote productivity means they require a progressively lower level of firm efficiency to make full remote work optimal. This effect shrinks the hybrid region from above.
    2.  **Threshold Gap Effect:** The term in the brackets, $\left[ 1 - \left( 1 - \frac{c_0}{A_1 h} \right)^{1/\nu} \right]$, represents the gap between the thresholds as a fraction of the upper threshold. This gap also shrinks as $h$ increases. Intuitively, as a worker's baseline productivity $A_1 h$ rises, the cost of being in the office ($c_0$) becomes smaller in relative terms, meaning the lower threshold moves closer to the upper threshold.
- **Case 2: Substitutability ($\phi < 0$):** The monotonicity of the hybrid region's size, $\Delta\psi(h)$, is **ambiguous** and depends on the specific parameter values.
	*   **Economic Intuition:** When skill and technology are substitutes, two competing economic forces are at play:
	    1.  **Upper Threshold Effect (Widening):** The upper threshold $\overline{\psi}(h)$ is now *increasing* in $h$. As a worker's skill increases, their relative remote productivity ($h^\phi$) falls. This makes it *harder* for them to qualify for the full-remote regime, which pushes the upper boundary outwards and tends to **widen** the hybrid region.
	    2.  **Threshold Gap Effect (Narrowing):** The relative gap between the thresholds continues to shrink as $h$ increases, for the same reason as in the complementarity case (the relative importance of $c_0$ diminishes). This effect tends to **narrow** the hybrid region.