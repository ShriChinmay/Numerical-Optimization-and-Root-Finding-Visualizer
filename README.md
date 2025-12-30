# Numerical-Optimization-and-Root-Finding-Visualizer
Numerical Methods Visualizer is a Python based interactive tool for visualizing classical root finding and optimization algorithms. It implements methods like Bisection, Regula-Falsi, Newton–Raphson, Golden Section, and Gradient Descent with input validation and proper handling of convergence and divergence.
----------------------------------------------
## Features

- Interactive CLI-driven workflow  
- Visualization of algorithm iterations step-by-step  
- Input sanitization and validation for user-defined functions  
- Proper handling of convergence, divergence, and invalid problem setups  
- Clear theoretical explanation before executing each algorithm  

---

## Implemented Algorithms

### Root Finding
- Bisection Method  
- Regula Falsi (False Position)  
- Newton–Raphson Method  

### Optimization
- Golden Section Search (1D optimization)  
- Gradient Descent (2D optimization)  

---

## Numerical Responsibility

The program intentionally detects and handles cases where algorithms **should not converge**, such as:
- Missing sign changes in bracketing methods  
- Zero or near-zero derivatives in Newton’s method  
- Saddle points and unbounded functions in gradient descent  
- Gradient explosion due to unsuitable step sizes  

Instead of returning misleading results, the program provides warnings or terminates safely.

---

## Tech Stack

- Python  
- NumPy  
- SymPy  
- Matplotlib  

---

## How to Run

1. Clone the repository
2. Navigate to the project directory
3. Run the program

---

## Intended Use

- Learning and understanding numerical methods
- Visualizing how algorithms behave step-by-step
- Educational demos for numerical analysis courses

---

## Notes

- This project emphasizes correctness over forced convergence.
- Some functions may intentionally fail to converge due to mathematical reasons (e.g. unbounded or non-convex functions).
- The project is designed as a learning and exploration tool rather than a production optimizer.

THIS PROJECT IS INTENDED FOR EDUCATIONAL USE
