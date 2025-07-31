### Derivation of the Optimal Work Arrangement

This appendix derives the optimal remote work policy, $\alpha^*(h, \psi)$, and analyzes the properties of the threshold functions that partition the labor market into distinct work arrangement regimes.

#### The Maximization Problem

Under the assumption of efficient bargaining, the firm and worker choose the remote work share $\alpha \in$ to maximize the total joint flow value of the match. With quasi-linear utility, this simplifies to maximizing the sum of output $Y(\alpha)$ minus the worker's in-office disutility $c(1-\alpha)$:
$$
\max_{\alpha \in [0,1]} \quad \Big\{Y(\alpha \mid h, \psi) - c(1-\alpha)\Big\}
$$
To solve this problem, we use the functional forms specified in the main text:

*   **Production Function:** $Y(\alpha \mid h, \psi) = A(h) \cdot \left[(1 - \alpha) + \alpha \cdot g(h, \psi)\right]$
*   **Baseline Productivity:** $A(h) = A_1 h$
*   **Relative Remote Productivity:** $g(h, \psi) = \nu\psi - \psi_0 + \phi\log(h)$
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
\alpha_{\text{interior}}(h, \psi) = 1 - \left[ \frac{A_1 h \left( 1 - g(h, \psi) \right)}{c_0} \right]^{\frac{1}{\chi}} = 1 - \left[ \frac{A_1 h \left( 1 - (\nu\psi - \psi_0 + \phi\log(h)) \right)}{c_0} \right]^{\frac{1}{\chi}}
$$

**Threshold Derivations**

The boundaries of the hybrid region are determined by analyzing the derivative of the surplus, $\frac{d(Y-c)}{d\alpha}$, at the corners $\alpha=0$ and $\alpha=1$.

1.  **Upper Threshold, $\overline{\psi}(h)$ (Boundary for Full Remote Work):** The market transitions to full remote work ($\alpha^*=1$) when the surplus is still increasing at $\alpha=1$. The boundary is where the derivative is zero:
    \begin{align*}
     A_1 h (g(h, \overline{\psi}(h)) - 1) + c_0(1-1)^\chi &= 0 \\
    A_1 h (g(h, \overline{\psi}(h)) - 1) &= 0 \\
    \nu\overline{\psi}(h) - \psi_0 + \phi\log(h) &= 1 \\
    \implies \overline{\psi}(h) &= \frac{1 + \psi_0 - \phi\log(h)}{\nu}
    \end{align*}

2.  **Lower Threshold, $\underline{\psi}(h)$ (Boundary for Full In-Person Work):** The market remains at full in-person work ($\alpha^*=0$) as long as the surplus is decreasing at $\alpha=0$. The boundary is where the derivative is zero:
    \begin{align*}
    A_1 h (g(h, \underline{\psi}(h)) - 1) + c_0(1-0)^\chi &= 0 \\
    A_1 h (g(h, \underline{\psi}(h)) - 1) &= -c_0 \\
    g(h, \underline{\psi}(h)) &= 1 - \frac{c_0}{A_1 h} \\
    \nu\underline{\psi}(h) - \psi_0 + \phi\log(h) &= 1 - \frac{c_0}{A_1 h} \\
    \implies \underline{\psi}(h) &= \frac{1 + \psi_0 - \phi\log(h) - \frac{c_0}{A_1 h}}{\nu}
    \end{align*}

#### Monotonicity and Properties

**Monotonicity of the Thresholds**

*   **Upper Threshold $\overline{\psi}(h)$:**
    $$
    \frac{\partial \overline{\psi}(h)}{\partial h} = \frac{\partial}{\partial h} \left( \frac{1 + \psi_0 - \phi\log(h)}{\nu} \right) = -\frac{\phi}{\nu h}
    $$
    Assuming skill-remote complementarity ($\phi > 0$), the upper threshold is **strictly decreasing** in worker skill. Higher-skilled workers require a lower level of firm efficiency to make full remote work optimal.

*   **Lower Threshold $\underline{\psi}(h)$:**
    $$
    \frac{\partial \underline{\psi}(h)}{\partial h} = \frac{\partial}{\partial h} \left( \frac{1 + \psi_0 - \phi\log(h) - \frac{c_0}{A_1 h}}{\nu} \right) = \frac{1}{\nu} \left( -\frac{\phi}{h} + \frac{c_0}{A_1 h^2} \right)
    $$
    The sign of this derivative depends on the relative strength of two competing effects:
    1.  The **skill-complementarity effect** ($-\frac{\phi}{h}$), which pushes the threshold down as higher skill makes remote work more attractive.
    2.  The **productivity effect** ($+\frac{c_0}{A_1 h^2}$), which pushes the threshold up. As skill $h$ increases, the opportunity cost of not working in the office (the baseline productivity $A_1 h$) becomes larger, making the firm more reluctant to offer remote work.

    To find the turning point, we set the derivative to zero:
    $$
    -\frac{\phi}{h} + \frac{c_0}{A_1 h^2} = 0 \implies \frac{c_0}{A_1 h^2} = \frac{\phi}{h} \implies h = \frac{c_0}{A_1 \phi}
    $$
    Let's define this critical skill level as $\hat{h} = \frac{c_0}{A_1 \phi}$.
    *   For $h < \hat{h}$, the productivity effect dominates, and $\frac{\partial \underline{\psi}(h)}{\partial h} > 0$. The threshold is **increasing**.
    *   For $h > \hat{h}$, the skill-complementarity effect dominates, and $\frac{\partial \underline{\psi}(h)}{\partial h} < 0$. The threshold is **decreasing**.

    Therefore, the lower threshold $\underline{\psi}(h)$ has an **inverted U-shape**, peaking at $h = \hat{h}$. This yields a rich sorting pattern: middle-skill workers are the most likely to be in hybrid arrangements, while both low-skill and high-skill workers are more likely to be in corner solutions (full in-person or full remote, respectively).