import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import re

def validate_1d_function(func_str):


    if not isinstance(func_str, str):
        return False, None, "Input must be a string"

    s = func_str.strip()

    s = re.sub(r"\s+", "", s)

    s = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', s)

    s = s.replace("^", "**")

    replacements = {
    "ln": "log",
    "Sin": "sin",
    "Cos": "cos",
    "Tan": "tan",
    "Exp": "exp",
    "Log": "log"
}


    for k, v in replacements.items():
        s = re.sub(rf"\b{k}\b", v, s, flags=re.IGNORECASE)

    x = sp.symbols('x')

    try:
        expr = sp.sympify(s)
    except Exception:
        return False, None, "Invalid mathematical expression"

    symbols = expr.free_symbols

    if len(symbols) == 0:
        return False, None, "Expression must contain variable x"

    if len(symbols) > 1:
        return False, None, "Only single-variable functions are allowed"

    sym = symbols.pop()
    if sym != x:
        return False, None, f"Invalid variable '{sym}'. Use only x"

    try:
        f = sp.lambdify(x, expr, 'numpy')
        f(1.0)
    except Exception:
        return False, None, "Function cannot be evaluated numerically"

    return True, s, "Valid function"

def validate_1d_or_2d_function(func_str):


    if not isinstance(func_str, str):
        return False, None, None, "Input must be a string"

    s = func_str.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("^", "**")

    replacements = {
        "Sin": "sin",
        "Cos": "cos",
        "Tan": "tan",
        "Exp": "exp",
        "Log": "log",
        "ln": "log"
    }
    s = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', s)
    
    for k, v in replacements.items():
        s = re.sub(rf"\b{k}\b", v, s, flags=re.IGNORECASE)

    x, y = sp.symbols('x y')

    try:
        expr = sp.sympify(s)
    except Exception:
        return False, None, None, "Invalid mathematical expression"

    symbols = expr.free_symbols

    if len(symbols) == 0:
        return False, None, None, "Expression must contain at least one variable"

    if len(symbols) > 2:
        return False, None, None, "At most two variables (x, y) are allowed"

    allowed = {x, y}
    if not symbols.issubset(allowed):
        bad = symbols - allowed
        return False, None, None, f"Invalid variable(s): {bad}. Use only x and y"

    dimension = len(symbols)

    try:
        if dimension == 1:
            f = sp.lambdify(x, expr, 'numpy')
            f(1.0)
        else:
            f = sp.lambdify((x, y), expr, 'numpy')
            f(1.0, 1.0)
    except Exception:
        return False, None, None, "Function cannot be evaluated numerically"

    return True, s, dimension, "Valid function"

THEORY = {

    "bisection": """
Bisection Method (Root Finding)

The bisection method is a bracketing technique used to find roots of a
continuous function.

If f(a) and f(b) have opposite signs, i.e.
f(a) * f(b) < 0,
then at least one root exists in the interval [a, b].

At each iteration:
1. Compute the midpoint c = (a + b) / 2
2. Replace the endpoint having the same sign as f(c)
3. Repeat until convergence

Properties:
- Guaranteed convergence
- Linear (slow) convergence
- Does not require derivatives
- Requires a sign change

Best used when reliability is more important than speed.

REQUIRES TWO POINTS a AND b, f(a)*f(b)<0 MUST HOLD. DO YOU HAVE SUCH POINTS? 
(DO YOU KNOW THE INTERVAL BETWEEN WHICH ROOTS LIE?) IF NO, PLEASE CHOOSE NEWTON-RAPHSON ALGORITHM!
""",

    "regula_falsi": """
Regula Falsi (False Position Method)

Regula Falsi is an improvement over bisection that uses linear interpolation
instead of the midpoint.

Given f(a) < 0 and f(b) > 0, the next approximation is:
c = (a*f(b) - b*f(a)) / (f(b) - f(a))

The interval is updated based on the sign of f(c), preserving the root bracket.

Properties:
- Guaranteed convergence
- Faster than bisection in many cases
- Linear convergence
- Does not require derivatives

Limitation:
- One endpoint may remain fixed for many iterations (slow progress).

REQUIRES TWO POINTS a AND b, f(a)*f(b)<0 MUST HOLD. DO YOU HAVE SUCH POINTS? 
(DO YOU KNOW THE INTERVAL BETWEEN WHICH ROOTS LIE?) IF NO, PLEASE CHOOSE NEWTON-RAPHSON ALGORITHM!
""",

    "newton": """
Newton-Raphson Method (Root Finding)

The Newton-Raphson method uses tangents to rapidly approximate a root.

Starting from an initial guess x0:
x_{n+1} = x_n - f(x_n) / f'(x_n)

Geometrically, this is the x-intercept of the tangent at x_n.

Properties:
- Very fast (quadratic convergence)
- Requires first derivative
- Sensitive to initial guess
- May fail if derivative is zero or small

Best used when a good initial guess and derivative are available.
""",

    "golden": """
Golden Section Method (Optimization)

The golden section method finds the minimum or maximum of a unimodal
one-dimensional function without using derivatives.

Two interior points are chosen using the golden ratio:
phi = (sqrt(5) - 1) / 2

The interval is reduced by comparing function values and reusing evaluations.

Properties:
- Derivative-free optimization
- Guaranteed convergence for unimodal functions
- Efficient function evaluations
- Linear convergence

Used for 1D optimization when derivatives are unavailable or unreliable.
This algorithm may behave wierdly if the function is not unimodal (contains more than 1 minima/maxima) in the given range
""",

    "gradient": """
Gradient Descent Method (Multivariable Optimization)

Gradient descent iteratively moves in the direction of steepest descent
of a function.

Update rule:
x_{n+1} = x_n - alpha * df/dx
y_{n+1} = y_n - alpha * df/dy

where alpha is the step size (learning rate).

Stopping criterion:
||grad f|| -> 0

Properties:
- Simple and general
- Requires gradients
- Sensitive to step size
- May converge slowly or zig-zag

Widely used in optimization and machine learning.
"""
}

def parse(func_str):
    x = sp.symbols('x')
    locals_dict = {
        'Sin': sp.sin,
        'Cos': sp.cos,
        'Tan': sp.tan,
        'Exp': sp.exp,
        'Log': sp.log
    }
    f_expr = sp.sympify(func_str)
    dfx_expr = sp.diff(f_expr, x)
    f = sp.lambdify(x, f_expr, 'numpy')
    dfx = sp.lambdify(x, dfx_expr, 'numpy')
    return f, dfx

def parse_2d(func_str):
    x, y = sp.symbols('x y')
    f_expr = sp.sympify(func_str)
    dfx_expr = sp.diff(f_expr, x)
    dfy_expr = sp.diff(f_expr, y)
    f = sp.lambdify((x, y), f_expr, 'numpy')
    dfx = sp.lambdify((x, y), dfx_expr, 'numpy')
    dfy = sp.lambdify((x, y), dfy_expr, 'numpy')
    return f, dfx, dfy

def bisection (f, a, b):
    if (f(a)>0):
        a,b=b,a
    xneg, yneg, xpos, ypos, nextx, nexty=[],[],[],[], [], []
    fa=f(a)
    fb=f(b)
    xneg.append(a)
    yneg.append(fa)
    xpos.append(b)
    ypos.append(fb)
    c=(a+b)/2
    fc=f(c)
    maxitr=1000
    itr=0
    converged=True
    while (abs(fc)>1e-3):
        if (maxitr<=0):
            raise ValueError("Failed to converge")
        c=(a+b)/2
        fc=f(c)
        if (fc<=0):
            a=c
            xneg.append(c)
            yneg.append(fc)
            xpos.append(xpos[-1])
            ypos.append(ypos[-1])
            nextx.append(c)
            nexty.append(fc)
        else:
            b=c
            xpos.append(c)
            ypos.append(fc)
            xneg.append(xneg[-1])
            yneg.append(yneg[-1])
            nextx.append(c)
            nexty.append(fc)
        
        fa=f(a)
        fb=f(b)

        itr+=1
        maxitr-=1
    if (abs(fc)>1e-3):
        converged=False
    return (c, fc, itr, xneg, yneg, xpos, ypos, nextx, nexty, converged)

def newton_raphson(f, dfx, a):
    xlist, ylist=[], []
    fa=f(a)
    itr=0
    converged=True
    xlist.append(a)
    ylist.append(fa)
    itrmax=100
    while(abs(fa)>1e-3 ):
        if abs(dfx(a)) < 1e-8:
            raise ValueError("Derivative too small; Newton fails")
        if itrmax<0:
            break
        b=a-fa/(dfx(a))
        xlist.append(b)
        ylist.append(f(b))
        a=b
        fa=f(b)
        itrmax-=1
        itr+=1
    if(abs(fa)>1e-3):
        converged=False
        
    return a, fa, itr, xlist, ylist, converged

def regula_falsi(f, a, b):
    if (f(a)>0):
        a,b=b,a
    fa=f(a)
    fb=f(b)
    itr=0
    xneg, yneg, xpos, ypos, nextx, nexty=[],[],[],[], [], []
    xneg.append(a)
    yneg.append(fa)
    xpos.append(b)
    ypos.append(fb)
    converged=True
    c=(a*fb-b*fa)/(fb-fa)
    fc=f(c)
    maxitr=1000
    while(abs(fc)>1e-3):
        if (maxitr<=0):
            break
        c=(a*fb-b*fa)/(fb-fa)

        fc=f(c)

        
        if(fc<0):
            a=c
            fa=fc
            xneg.append(c)
            yneg.append(fc)
            xpos.append(xpos[-1])
            ypos.append(ypos[-1])
            nextx.append(c)
            nexty.append(fc)
        else:
            b=c
            fb=fc
            xpos.append(c)
            ypos.append(fc)
            xneg.append(xneg[-1])
            yneg.append(yneg[-1])
            nextx.append(c)
            nexty.append(fc)
        itr+=1
        maxitr-=1
    if(abs(fc)>1e-3):
        converged=False
    return (c, fc, itr, xneg, yneg, xpos, ypos,nextx, nexty, converged)

def golden_section(f, a, b, opt):
    # only for unimodal intervals
    g = (5**0.5 - 1) / 2
    itr = 0

    left, right = [a], [b]

    x1 = b - g * (b - a)
    x2 = a + g * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    while (b - a) > 1e-3:
        if opt == 1:  # minimization
            cond = f1 >= f2
        elif opt == 2:  # maximization
            cond = f1 <= f2
        else:
            raise ValueError("opt must be 1 (min) or 2 (max)")

        if cond:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + g * (b - a)
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - g * (b - a)
            f1 = f(x1)

        left.append(a)
        right.append(b)
        itr += 1

    xmin = (a + b) / 2
    return xmin, f(xmin), itr, left, right

def gradient_descent(f, dfx, dfy, a, step=0.01):
    xlist, ylist, flist=[], [], []
    x0, y0=a
    itr=0
    for i in range (100000):
        
        x1=x0-step*dfx(x0, y0)
        y1=y0-step*dfy(x0, y0)
        x0, y0=x1, y1
        xlist.append(x1)
        ylist.append(y1)
        flist.append(f(x1, y1))
        itr+=1
        gx = dfx(x0, y0)
        gy = dfy(x0, y0)

        grad_norm = (gx*gx + gy*gy)**0.5

        if grad_norm > 1e6:
            raise ValueError("Gradient exploded. Function may be unbounded or step too large. \nTHIS ALGORITHM MIGHT NOT BE SUITABLE FOR THE OPTIMISASTION OF GIVEN FUNCTION!")

        if grad_norm < 1e-3:
            break
    return (x0, y0), f(x0, y0), itr, xlist, ylist, flist

fncs=("(x-1)*(x-5)")
def root_finder1(funcn_str, left, right, choice, domleft=-100, domright=100):
    fx, dfx=parse(funcn_str)
    if fx(left)*fx(right)>0:
        print("Invalid endpoints")
        return
    xs=np.linspace(domleft, domright, abs(domright-domleft)*10)
    ys=fx(xs)
    fig, ax=plt.subplots(figsize=(8, 5), dpi=120)
    fig.patch.set_facecolor("#F8F8F2")
    ax.set_facecolor("#F8F8F2")
    ax.grid(
        True,
        which='both',
        linestyle='--',
        linewidth=0.6,
        alpha=0.4
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', colors='#333333')
    ax.plot(xs, ys, color='#222222', linewidth=1.6, label='f(x)')
    ax.plot(xs, np.zeros_like(xs), color='black', linewidth=1)
    negplot, =ax.plot([], [],'o', markersize=4, color='#D62728', label='Current Interval Endpoints')
    posplot, =ax.plot([], [],'o', markersize=4, color='#D62728')
    negvertplot, =ax.plot([], [], linewidth=1, color='#A6A6A6', alpha=0.7)
    posvertplot, =ax.plot([], [], linewidth=1, color='#A6A6A6', alpha=0.7)
    horplot, =ax.plot([], [], color='#555555', linewidth=1.5, linestyle='--')
    midvertplot, =ax.plot([], [], linewidth=1.5, color='#FF7F0E', alpha=0.8)
    midptplot, = ax.plot([], [],'o',color='#FF7F0E',markersize=5,markeredgecolor='black',markeredgewidth=0.5,label='Current root')
    finplot, =ax.plot([], [], markersize=10, marker='X',color="#200096", label='Final Root', linestyle='None')
    ax.set_xlim(left-1, right+1)
    lo=min(fx(left), fx(right))
    hi=max(fx(left), fx(right))
    ax.set_ylim(lo-0.1*abs(lo), hi+0.1*abs(hi))
    if choice==1:
        a, fa, itr, xn, yn,xp, yp, nextx, nexty, cnv=bisection(fx, left, right)
    elif choice==2:
        a, fa, itr, xn, yn,xp, yp, nextx, nexty, cnv=regula_falsi(fx, left, right)
    print("Root lies at x=", round(a,4) , "\nRoot found out in ", itr, "iterations")
    ax.legend(frameon=False, loc='upper right')

    def update_root1(frame):
        negvertplot.set_data([], [])
        posvertplot.set_data([], [])
        horplot.set_data([], [])
        midvertplot.set_data([], [])
        midptplot.set_data([], [])
        
        if frame==4*len(xn):
            midptplot.set_data([], [])
            if nextx:
                finplot.set_data([nextx[-1]], [nexty[-1]])
            else:
                finplot.set_data([a], [fa])
            negplot.set_data([], [])
            posplot.set_data([], [])
        elif (frame%4==0):
            negplot.set_data(xn[(frame//4):(frame//4)+1], yn[(frame//4):(frame//4)+1])
            posplot.set_data(xp[(frame//4):(frame//4)+1], yp[(frame//4):(frame//4)+1])
        elif (frame%4==1):
            negplot.set_data(xn[(frame//4):(frame//4)+1], yn[(frame//4):(frame//4)+1])
            posplot.set_data(xp[(frame//4):(frame//4)+1], yp[(frame//4):(frame//4)+1])
            negvertplot.set_data(np.linspace(xn[(frame//4)], xn[(frame//4)], 1000), np.linspace(-100, 100, 1000))
            posvertplot.set_data(np.linspace(xp[(frame//4)], xp[(frame//4)], 1000), np.linspace(-100, 100, 1000))
        elif (frame%4==2):
            negplot.set_data(xn[(frame//4):(frame//4)+1], yn[(frame//4):(frame//4)+1])
            posplot.set_data(xp[(frame//4):(frame//4)+1], yp[(frame//4):(frame//4)+1])
            negvertplot.set_data(np.linspace(xn[(frame//4)], xn[(frame//4)], 1000), np.linspace(-100, 100, 1000))
            posvertplot.set_data(np.linspace(xp[(frame//4)],xp[(frame//4)], 1000), np.linspace(-100, 100, 1000))
            horplot.set_data(np.linspace(-100, 100, 1000), np.linspace(yp[(frame//4)], yp[(frame//4)], 1000))   
        elif (frame%4==3):
            negplot.set_data(xn[(frame//4):(frame//4)+1], yn[(frame//4):(frame//4)+1])
            posplot.set_data(xp[(frame//4):(frame//4)+1], yp[(frame//4):(frame//4)+1])
            negvertplot.set_data(np.linspace(xn[(frame//4)], xn[(frame//4)], 1000), np.linspace(-100, 100, 1000))
            posvertplot.set_data(np.linspace(xp[(frame//4)],xp[(frame//4)], 1000), np.linspace(-100, 100, 1000))
            horplot.set_data(np.linspace(-100, 100, 1000),  np.linspace(yp[(frame//4)], yp[(frame//4)], 1000))   
            if frame//4 <len(nextx):
                midptplot.set_data([nextx[frame//4]], [nexty[frame//4]])    
                midvertplot.set_data(np.linspace(nextx[frame//4], nextx[frame//4], 1000), np.linspace(-100, 100, 1000)) 
                 
        

        return negplot, posplot, negvertplot, posvertplot, horplot, midptplot
    animplot=FuncAnimation(fig=fig, frames=4*len(xn)+1, interval=1000, repeat=False, func=update_root1)
    plt.show()

def root_finder2(funcn_str, x0):
    fx, dfx=parse(funcn_str)
    if abs(dfx(x0)) < 1e-8:
        raise ValueError("Derivative too small; Newton fails")
    a, fa, itr, xl, yl, cnv=newton_raphson(fx, dfx, x0)
    print("Root lies at x=", round(a, 4), "\nFound in ", itr, "iterations")
    fig, ax=plt.subplots(figsize=(8, 5), dpi=120)
    fig.patch.set_facecolor("#F8F8F2")
    ax.set_facecolor("#F8F8F2")
    ax.grid(
        True,
        which='both',
        linestyle='--',
        linewidth=0.6,
        alpha=0.4
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', colors='#333333')    
    ax.plot(np.linspace(-100, 100, 1000000), fx(np.linspace(-100, 100, 1000000)), color='#222222', linewidth=1.6, label='f(x)')
    ax.plot(np.linspace(-100, 100, 1000000), np.linspace(0, 0, 1000000), color='black', linewidth=1)
    lo = min(xl)
    hi = max(xl)
    ax.set_xlim(lo - 1, hi + 1)

    ymin = min(yl)
    ymax = max(yl)
    pad = 0.3 * (ymax - ymin + 1e-6)
    ax.set_ylim(ymin - pad, ymax + pad)
    currplot, =ax.plot([], [],  markersize=4, color='#D62728', marker='o', linestyle='None', label='Current Root')
    gradplot, =ax.plot([], [],  linewidth=1, color="#45B1CC", label='Tangent')
    vertplot, =ax.plot([], [])
    nextplot, =ax.plot([], [], 'o',color='#FF7F0E',markersize=5,markeredgecolor='black',markeredgewidth=0.5,label='Next root')
    finplot, =ax.plot([], [], markersize=10, marker='X',color="#200096", label='Final Root', linestyle='None')
    ax.legend(frameon=False, loc='upper right')
    def update(frame):
        gradplot.set_data([], [])
        currplot.set_data([], [])
        vertplot.set_data([], [])
        nextplot.set_data([], [])
        finplot.set_data([], [])
        fp=frame%4
        fr=frame//4
        if (fr<len(xl)):
            df=dfx(xl[fr])
            y0=yl[fr]
            x0=xl[fr]

        if (frame==4*len(xl)):
            finplot.set_data([xl[-1]], [yl[-1]])
        elif (fp==0):
            currplot.set_data([x0], [y0])
        elif fp==1:
            currplot.set_data([x0], [y0])
            xk=np.linspace(-100, 100, 1000)
            yk=df*xk-df*x0+y0
            gradplot.set_data([xk], [yk])
        elif fp==2:
            currplot.set_data([x0], [y0])
            xk=np.linspace(-100, 100, 1000)
            yk=df*xk-df*x0+y0
            gradplot.set_data([xk], [yk])
            xs=(0-y0)/(df) +x0
            vertplot.set_data([np.linspace(xs, xs, 1000)], [np.linspace(-100, 100, 1000)])
        else:
            currplot.set_data([x0], [y0])
            xk=np.linspace(-100, 100, 1000)
            yk=df*xk-df*x0+y0
            gradplot.set_data([xk], [yk])
            xs=(0-y0)/(df) +x0
            vertplot.set_data([np.linspace(xs, xs, 1000)], [np.linspace(-100, 100, 1000)])
            nextplot.set_data([xs], [fx(xs)])
        return currplot, gradplot, vertplot, nextplot
            
    animplot=FuncAnimation(fig=fig, frames=4*len(xl)+1, interval=1000, func=update, repeat=False)    

    plt.show()

def golden_sec_anim(funcn_str, left, right, opt):
    fx, _ = parse(funcn_str)
    xmin, fxmin, itr, lefts, rights = golden_section(fx, left, right, opt)
    print("Minimum value of the function was found out to be ", round(fxmin, 4), "at x=", round(xmin, 4), ".\nFound out in", itr, " iterations")
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#F8F8F2")
    ax.set_facecolor("#F8F8F2")
    ax.grid(
        True,
        which='both',
        linestyle='--',
        linewidth=0.6,
        alpha=0.4
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', colors='#333333')   
    
    xs = np.linspace(lefts[0] - 0.5 * abs(lefts[0]),rights[0] + 0.5 * abs(rights[0]), 1000)
    ys = fx(xs)
    ax.plot(xs, ys, color='black', lw=1.5)
    pad = 0.2 * (ys.max() - ys.min())
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ypt = ys.max() + 0.1 * (ys.max() - ys.min())
    horplot, = ax.plot(
        [xs.min(), xs.max()],
        [ypt, ypt],
        color='gray',
        linestyle='--'
    )
    leftplot, = ax.plot([], [], 'ro')
    rightplot, = ax.plot([], [], 'bo')
    finplot, =ax.plot([], [], 'bX', markersize=8)
    leftvertplot, = ax.plot([], [], 'r--', alpha=0.7)
    rightvertplot, = ax.plot([], [], 'b--', alpha=0.7)
    leftfunc, = ax.plot([], [], 'ro')
    rightfunc, = ax.plot([], [], 'bo')

    ymin, ymax = ax.get_ylim()

    def update(frame):
        if frame==len(lefts)+1:
            finplot.set_data([xmin], [fxmin])
            leftplot.set_data([], [])
            rightplot.set_data([], [])
            leftvertplot.set_data([], [])
            rightvertplot.set_data([], [])
            leftfunc.set_data([], [])
            rightfunc.set_data([], [])

        else:
            if frame == 0:
                return ()

            l = lefts[frame - 1]
            r = rights[frame - 1]
            leftplot.set_data([l], [ypt])
            rightplot.set_data([r], [ypt])
            leftvertplot.set_data([l, l], [ymin, ymax])
            rightvertplot.set_data([r, r], [ymin, ymax])
            leftfunc.set_data([l], [fx(l)])
            rightfunc.set_data([r], [fx(r)])

        return (leftplot, rightplot,leftvertplot, rightvertplot,leftfunc, rightfunc, finplot)

    anim = FuncAnimation(fig,update,frames=len(lefts) + 2,interval=800,blit=True,repeat=False)

    plt.show()

def gradient_descent_anim(funcn_str, a, step=0.01):
    f, dfx, dfy=parse_2d(funcn_str)
    (xmin, ymin), fmin, itr, xlist, ylist, flist = gradient_descent(
        f, dfx, dfy, a, step
    )
    print("Minimum value of the function was found out to be ", round(fmin, 4), "at (x, y)=(", round(xmin, 4),",", round(ymin, 4), ").\nFound out in", itr, " iterations")
    xlist = np.array(xlist)
    ylist = np.array(ylist)
    if (itr==100000):
        print("WARNING: Gradient Descent didn't converge, result might not be the true minimum")
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#F8F8F2")
    ax.set_facecolor("#F8F8F2")

    ax.grid(
        True,
        linestyle='--',
        linewidth=0.6,
        alpha=0.4
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    pad = 0.5
    x_min, x_max = xlist.min() - pad, xlist.max() + pad
    y_min, y_max = ylist.min() - pad, ylist.max() + pad

    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = f(X, Y)

    ax.contour(X, Y, Z, levels=30, cmap="viridis", alpha=0.8)

    pathplot, = ax.plot([], [], 'r-', lw=1.8, label="Path")
    pointplot, = ax.plot([], [], 'ro', label="Current")
    finplot, = ax.plot([], [], 'bX', markersize=9, label="Minimum")

    ax.legend()

    def update(frame):
        if frame == 0:
            return ()

        if frame == len(xlist):
            finplot.set_data([xmin], [ymin])
            pointplot.set_data([], [])
            return (finplot,)

        pathplot.set_data(xlist[:frame], ylist[:frame])
        pointplot.set_data([xlist[frame]], [ylist[frame]])

        return pathplot, pointplot

    anim = FuncAnimation(fig,update,frames=len(xlist) + 1, interval=50, blit=True,repeat=False)

    plt.show()


def choose1():
    print("-----------------------------------------------------------\n")
    print("Choose desired: \n1. Root Finding \n2. Optimisation (Minimisation/ Maximisation) \n  Enter 1 or 2")
    inp=int(input("Enter: "))
    if inp!=1 and inp!=2:
        print("Invalid entry")
        choose1()
    else:
        return inp

def choose_root_algo():
    print("-----------------------------------------------------------\n")
    print("Choose root finding algorithm \n1. Bisection algorithm \n2. Regula-falsi algorithm \n3. Newton-Raphson algortihm")
    inp=int(input("Enter: "))
    if inp!=1 and inp!=2 and inp!=3:
        print("Invalid Entry")
        choose_root_algo()
    else:
        return inp

def choose_opt_algo():
    print("-----------------------------------------------------------\n")
    print("Choose optimisation algorithm \n1. Golden Section algorithm \n2. Gradient Descent algorithm")
    inp=int(input("Enter: "))
    if inp!=1 and inp!=2:
        print("Invalid Entry")
        choose_opt_algo()
    else:
        return inp


def main():
    choice1=choose1()
    if choice1==1:
        
        root_algo=choose_root_algo()
        print("-----------------------------------------------------------\n")
        if root_algo==1:
            print(THEORY['bisection'])
        elif root_algo==2:
            print(THEORY['regula_falsi'])
        elif root_algo==3:
            print(THEORY['newton'])
        print("-----------------------------------------------------------\n")
        inp=input("Enter 1 to continue, any other character to go back: ")
        if inp!='1':
            main()
        else:
            print("-----------------------------------------------------------\n")
            func_str = input("Enter function f(x): ")
            valid, clean_func, msg = validate_1d_function(func_str)
            if not valid:
                print("Error:", msg)
                choose_root_algo()
            print("Accepted function:", clean_func)
            if root_algo==1 or root_algo==2:
                print("-----------------------------------------------------------\n")
                left=float(input("Enter the left endpoint of the interval: "))
                right=float(input("Enter the right endpoint of the interval: "))
                root_finder1(clean_func, left, right, root_algo)
            else:
                print("-----------------------------------------------------------\n")
                x0=float(input("Enter starting point: "))
                root_finder2(clean_func, x0)

    elif choice1==2:
        
        opt_algo=choose_opt_algo()
        print("-----------------------------------------------------------\n")
        if opt_algo==1:
            print(THEORY['golden'])
        elif opt_algo==2:
            print(THEORY['gradient'])
        print("-----------------------------------------------------------\n")
        inp=input("Enter 1 to continue, any other character to go back: ")
        if inp!='1':
            main()
        else:
            if opt_algo==1:
                print("-----------------------------------------------------------\n")
                func_str = input("Enter function f(x): ")
                valid, clean_func, msg = validate_1d_function(func_str)
                if not valid:
                    print("Error:", msg)
                    choose_root_algo()
                print("Accepted function:", clean_func)
                print("-----------------------------------------------------------\n")
                left=float(input("Enter the left endpoint of the interval: "))
                right=float(input("Enter the right endpoint of the interval: "))
                opt=int(input("Enter 1 for minimisation, 2 for maximisation: "))
                if opt!=1 and opt!=2:
                    print("Error")
                    return
                golden_sec_anim(clean_func, left, right, opt)
            
            elif opt_algo==2:
                print("-----------------------------------------------------------\n")
                func_str = input("Enter 2D function f(x, y): ")
                valid, clean_func, dim, msg = validate_1d_or_2d_function(func_str)
                if not valid:
                    print("Error:", msg)
                    choose_root_algo()
                print("Accepted function:", clean_func)
                print("-----------------------------------------------------------\n")
                x0=float(input("Enter x coordinate of starting point: "))
                y0=float(input("Enter y coordinate of starting point: "))
                gradient_descent_anim(clean_func, (x0, y0))


main()