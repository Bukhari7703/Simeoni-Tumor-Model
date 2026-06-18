import sympy as sp

# ==========================================
# 1. Define Mathematical Symbols
# ==========================================
# State variables
z0, z1, z2, z3, q = sp.symbols('z0 z1 z2 z3 q', real=True)

# Control variable
u = sp.symbols('u', real=True)

# Co-state variables (Lagrange multipliers)
lam_z0, lam_z1, lam_z2, lam_z3, lam_q = sp.symbols('lambda_z0 lambda_z1 lambda_z2 lambda_z3 lambda_q', real=True)

# Biological Parameters
l0, l1, psi = sp.symbols('lambda0 lambda1 psi', positive=True) # Growth parameters (L0=lambda0, L1=lambda1)
k1, k2, k10, v = sp.symbols('k1 k2 k10 v', positive=True)      # Pharmacokinetic rates
Rs = sp.symbols('Rs', positive=True)                           # Cost weighting

# ==========================================
# 2. Build the System Dynamics (ODEs)
# ==========================================
# Total weight w(t)
w = z0 + z1 + z2 + z3

# The non-linear growth term
growth_term = (l0 * z0) * ( 1 + (l0/l1 * w)**psi )**(-1/psi)

# State Equations (f)
# dz0 = growth_term - k2 * (q/v) * z0
dz0 = (l0 * z0) * ( 1 + (l0/l1 * w)**psi )**(-1/psi) - k2 * (q/v) * z0
dz1 = k2 * (q/v) * z0 - k1 * z1
dz2 = k1 * (z1 - z2)
dz3 = k1 * (z2 - z3)
dq  = -k10 * q + u

# ==========================================
# 3. Construct the Hamiltonian (H)
# ==========================================
# Cost Integrand (L)
Cost = 0.5 * (1000 * (z0**2 + z1**2 + z2**2 + z3**2) + 0.04 * q**2 + Rs * u**2)

# H = Cost + Sum(lambda * f)
H = Cost + lam_z0*dz0 + lam_z1*dz1 + lam_z2*dz2 + lam_z3*dz3 + lam_q*dq

# ==========================================
# 4. Calculate Co-State Equations (-dH/dx)
# ==========================================
# Take the partial derivatives and multiply by -1
dlam_z0 = -sp.diff(H, z0)
dlam_z1 = -sp.diff(H, z1)
dlam_z2 = -sp.diff(H, z2)
dlam_z3 = -sp.diff(H, z3)
dlam_q  = -sp.diff(H, q)

# ==========================================
# 5. Print the Results
# ==========================================
print("--- CO-STATE EQUATIONS ---\n")

print("1. d(lambda_z0)/dt =")
sp.pprint(sp.simplify(dlam_z0))
print("\n")

print("2. d(lambda_z1)/dt =")
sp.pprint(sp.simplify(dlam_z1))
print("\n")

print("3. d(lambda_z2)/dt =")
sp.pprint(dlam_z2)
print("\n")

print("4. d(lambda_z3)/dt =")
sp.pprint(dlam_z3)
print("\n")

print("5. d(lambda_q)/dt =")
sp.pprint(dlam_q)

# Create a dummy symbol for 'w' just for printing
w_sym = sp.symbols('w')

print("--- CLEANED CO-STATE EQUATIONS ---\n")

# 1. Substitute the expanded z0+z1+z2+z3 back into 'w' to make it readable
clean_z1 = sp.simplify(dlam_z1).subs(z0 + z1 + z2 + z3, w_sym)
clean_z2 = sp.simplify(dlam_z2).subs(z0 + z1 + z2 + z3, w_sym)
clean_z3 = sp.simplify(dlam_z3).subs(z0 + z1 + z2 + z3, w_sym)

print("2. d(lambda_z1)/dt =")
sp.pprint(clean_z1)
print("\n")

print("3. d(lambda_z2)/dt =")
sp.pprint(clean_z2)
print("\n")

print("4. d(lambda_z3)/dt =")
sp.pprint(clean_z3)
print("\n")

print("5. d(lambda_q)/dt =")
# This one is already pretty clean!
sp.pprint(sp.simplify(dlam_q))