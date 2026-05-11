import matplotlib.pyplot as plt
import numpy as np
import os
import time


#############
# Methods
#############

# Exact solution (Const T_amb)
def exact_c(T0, k, T_amb, t_start, t_end, n_steps):
    # Temperature List
    print(f"(k = {k})")
    total_steps = n_steps + 1 # bcs we also count time = 0.0sec
    T = [0.0] * total_steps

    # Time
    t_n = [0.0] * total_steps
    t_n[0] = t_start
    h = (t_end - t_start)/n_steps

    for n in range(total_steps): # n_steps -1: bcs T0 is previous defined
        t_n[n] = t_start + n*h
        T[n] = T_amb + (T0 - T_amb)/(np.e**(k*t_n[n]))
        print(f"t = {t_n[n]:.2f}        T[{n}] = {T[n]:.2f}")

    return T, t_n


# Exact solution (Variable T_amb)
def exact_v(T0, k, t_start, t_end, n_steps):
    # Temperature List
    print(f"(k = {k})")
    total_steps = n_steps + 1 # bcs we also count time = 0.0sec
    T = [0.0] * total_steps

    # Time
    t_n = [0.0] * total_steps
    t_n[0] = t_start
    h = (t_end - t_start)/n_steps

    for n in range(total_steps): # n_steps -1: bcs T0 is previous defined
        t_n[n] = t_start + n*h
        T[n] = 82 - (10*k)/(k**2 + (np.pi/12)**2) * (k * np.sin(np.pi/12 * t_n[n] + np.pi/4) - (np.pi/12)*np.cos((np.pi/12)*t_n[n] + np.pi/4)) + (T0 - 82 + (5*np.sqrt(2)*k*(k - np.pi/12)/(k**2 + (np.pi/12)**2)))*np.exp(-k*t_n[n])
        print(f"t = {t_n[n]:.2f}        T[{n}] = {T[n]:.2f}")

    return T, t_n



# f(t,T) = -k(T-T_{amb})
def cooling_ode(t, T, k, T_amb):
    return -k * (T - T_amb)

# Explicit Euler method (Const T_amb)
def explicit_c(f, T0, k, T_amb, t_start, t_end, n_steps):
    # T(n+1) = T(n) + h \cdot (f(t_n, T_n ))
    # f: derivate function or slope f(t,T)
    print(f"(k = {k})")
    h = (t_end - t_start)/n_steps # size of each step

    total_steps = n_steps + 1 # bcs we also count time = 0.0sec

    # Temperature List
    T = [0.0] * total_steps
    T[0] = T0 # Initial temperature

    # Time
    t_n = [0.0] * total_steps
    t_n[0] = t_start

    print(f"t = {t_n[0]:.2f}        T[0] = {T[0]:.2f}")

    # Explicit euler
    for n in range(total_steps-1): # n_steps -1: bcs T0 is previous defined
        t_n[n+1] = t_n[n] + h
        T[n+1] = T[n] + h * f(t_n[n], T[n], k, T_amb)
        print(f"t = {t_n[n+1]:.2f}      T[{n+1}] = {T[n+1]:.2f}")

    return T,t_n


# Explicit Euler method (Variable T_amb)
def explicit_v(T0, k, T_amb_func, t_start, t_end, n_steps):
    # T(n+1) = T(n) + h \cdot (f(t_n, T_n ))
    # f: derivate function or slope f(t,T)
    print(f"(k = {k})")
    h = (t_end - t_start)/n_steps # size of each step

    total_steps = n_steps + 1 # bcs we also count time = 0.0sec

    # Temperature List
    T = [0.0] * total_steps
    T[0] = T0 # Initial temperature

    # Time
    t_n = [0.0] * total_steps
    t_n[0] = t_start

    print(f"t = {t_n[0]:.2f}        T[0] = {T[0]:.2f}")

    # Explicit euler
    for n in range(total_steps-1): # n_steps -1: bcs T0 is previous defined
        t_n[n+1] = t_n[n] + h
        T[n+1] = T[n] + h * k * (T_amb_func(t_n[n]) - T[n])
        print(f"t = {t_n[n+1]:.2f}      T[{n+1}] = {T[n+1]:.2f}")

    return T,t_n



# Implicit Euler method (Const T_amb)
def implicit_c(T0, k, T_amb, t_start, t_end, n_steps):
    # f: derivate function or slope f(t,T)
    print(f"(k = {k})")
    h = (t_end - t_start)/n_steps # size of each step

    total_steps = n_steps + 1 # bcs we also count time = 0.0sec

    # Temperature List
    T = [0.0] * total_steps
    T[0] = T0 # Initial temperature

    # Time
    t_n = [0.0] * total_steps
    t_n[0] = t_start

    print(f"t = {t_n[0]:.2f}        T[0] = {T[0]:.2f}")

    # Implicit euler
    for n in range(total_steps-1): # n_steps -1: bcs T0 is previous defined
        t_n[n+1] = t_n[n] + h
        T[n+1] = (T[n] + h * k * T_amb)/(1 + h*k)
        print(f"t = {t_n[n+1]:.2f}      T[{n+1}] = {T[n+1]:.2f}")

    return T,t_n

def implicit_v(T0, k, T_amb_func, t_start, t_end, n_steps):
    # f: derivate function or slope f(t,T)
    print(f"(k = {k})")
    h = (t_end - t_start)/n_steps # size of each step

    total_steps = n_steps + 1 # bcs we also count time = 0.0sec

    # Temperature List
    T = [0.0] * total_steps
    T[0] = T0 # f, T0, k, T_amb, t_start, t_end, n_steps)Initial temperature

    # Time
    t_n = [0.0] * total_steps
    t_n[0] = t_start

    print(f"t = {t_n[0]:.2f}        T[0] = {T[0]:.2f}")

    # Implicit euler
    for n in range(total_steps-1): # n_steps -1: bcs T0 is previous defined
        t_n[n+1] = t_n[n] + h
        T[n+1] = (T[n] + h * k * T_amb_func(t_n[n+1]))/(1 + h*k)
        print(f"t = {t_n[n+1]:.2f}      T[{n+1}] = {T[n+1]:.2f}")

    return T,t_n



def global_error(T_exact_const, T_method): # Assuming both with the same time t
    T_ex = np.array(T_exact_const)
    T_m = np.array(T_method)

    error = np.max(np.abs(T_ex - T_m))

    return error


def print_parameters(T0, T_amb, k):
    print(f"""(Initial temperature) T0 = {T0}
(Ambient temperature) T_amb = {T_amb}
(Cooling constant) k = {k}\n""")


#########
# Plots
#########

def plot_temperature(T, t_n, method_name, color):

    # Try change color=".." to a static variable relative to each experiment
    plt.plot(t_n, T, color=color, label=method_name)
    plt.title(method_name)
    plt.xlabel("Time (min)")
    plt.ylabel("Temperature (ºC)")

    os.makedirs("plots", exist_ok=True)
    #plt.legend()

def plot_convergence(h_l, exp_error, impl_error, num_h, k_n, cons_or_var):
    for j, k in enumerate(k_n):
        exp_err_k = [exp_error[i][j] for i in range(num_h)]
        imp_err_k = [impl_error[i][j] for i in range(num_h)]

        # Create ONE figure
        plt.figure(figsize=(7, 5))

        # Plot both methods
        plt.loglog(h_l, exp_err_k, 'o-', label=f'Explicit Euler')
        plt.loglog(h_l, imp_err_k, 's--', label=f'Implicit Euler')

        # Add reference line of slope 1 (using the first data point of explicit)
        C = exp_err_k[0] / h_l[0]   # E =~ C * h¹
        plt.loglog(h_l, [C * h for h in h_l], 'k:', label='Slope 1')

        # Labels and title
        plt.xlabel('Step size h (min)')
        plt.ylabel('Global error E(h)')
        plt.title(f'Convergence study (k = {k:.2f})_{cons_or_var}')
        plt.legend()
        plt.grid(True)

        # Save and show
        plt.savefig(f'plots/convergence_k{k:.2f}_{cons_or_var}.pdf')
        plt.show()


def save_plots(title, name):
    plt.legend()
    plt.title(title)
    plt.savefig(f"plots/{name}.pdf")
    plt.show()



##########
# Main
##########

def main():
    print("Newton Cooling Simulation")

    ############
    # Variables
    ############

    T0 = 200.0      # Initial temperature T(0)
    T_amb = 70.0    # Ambient temperature constant
    T_amb_func = lambda t: 82 - 10*np.sin(((2*np.pi)*(t+3))/24) # Ambient temperature variable

    k_start = 0.03
    k_end = 0.3
    #k_step = 0.05
    k_step = 1
    k_n = [i for i in np.arange(k_start, k_end, k_step)] # Cooling constant | k \in [0.03, 0.3]min^{-1}


    t_start = 0.0   # start time    (min)
    t_end = 60.0    # end time      (min)
    #n_steps = 100   # number of parts to be divided
    

    h_list = [20, 10, 5, 2.0, 0.5, 0.1, 0.05]   # in minutes


    # Pre‑allocate: rows = number of h values, columns = number of k values
    num_h = len(h_list)
    num_k = len(k_n)

    # Ambient Constant
    # Analytical (Ambient constant)
    T_exa_c = [[None]*num_k for _ in range(num_h)]      # Temperature Exact
    t_exa_c = [[None]*num_k for _ in range(num_h)]      # Time Exact 
    cont_exa_c = [[None]*num_k for _ in range(num_h)]   # Time counter 

    # Method Explicit Euler (Ambient constant)
    T_exp_c = [[None]*num_k for _ in range(num_h)]      # Temperature Explicit Euler
    t_exp_c = [[None]*num_k for _ in range(num_h)]      # Time Explicit Euler 
    err_exp_c = [[None]*num_k for _ in range(num_h)]    # Error Explicit Euler
    cont_exp_c = [[None]*num_k for _ in range(num_h)]   # Time counter 

    # Method Implicit Euler (Ambient constant)
    T_imp_c = [[None]*num_k for _ in range(num_h)]      # Temperature Implicit Euler
    t_imp_c = [[None]*num_k for _ in range(num_h)]      # Time Implicit
    err_imp_c = [[None]*num_k for _ in range(num_h)]    # Error Explicit Euler
    cont_imp_c = [[None]*num_k for _ in range(num_h)]   # Time counter 


    # Ambient Variable
    # Analytical (Ambient Variable)
    T_exa_v = [[None]*num_k for _ in range(num_h)]      # Temperature Exact
    t_exa_v = [[None]*num_k for _ in range(num_h)]      # Time Exact 
    cont_exa_v = [[None]*num_k for _ in range(num_h)]   # Time counter 

    # Method Explicit Euler (Ambient Variable)
    T_exp_v = [[None]*num_k for _ in range(num_h)]      # Temperature Explicit Euler
    t_exp_v = [[None]*num_k for _ in range(num_h)]      # Time Explicit Euler 
    err_exp_v = [[None]*num_k for _ in range(num_h)]    # Error Explicit Euler
    cont_exp_v = [[None]*num_k for _ in range(num_h)]   # Time counter 

    # Method Implicit Euler (Ambient Variable)
    T_imp_v = [[None]*num_k for _ in range(num_h)]      # Temperature Implicit Euler
    t_imp_v = [[None]*num_k for _ in range(num_h)]      # Time Implicit
    err_imp_v = [[None]*num_k for _ in range(num_h)]    # Error Explicit Euler
    cont_imp_v = [[None]*num_k for _ in range(num_h)]   # Time counter 



    for i, h in enumerate(h_list):
        print(f"------------- h = {h} -------------")
        n_steps = int((t_end - t_start) / h) # Number of parts to be divided

        print("Exact Solution, (Temperature Ambient CONSTANT)")
        # Using the explicit euler for each k
        for j, k in enumerate(k_n):
            cont_start = time.perf_counter()
            T_exa_c[i][j], t_exa_c[i][j] = exact_c(T0, k, T_amb, t_start, t_end, n_steps)
            cont_end = time.perf_counter()

            cont_exa_c[i][j] = cont_end - cont_start # Time for each solution (k * h)
            print(f"Time taken: {cont_exa_c[i][j]} seconds")

            print("\n")
            plot_temperature(T_exa_c[i][j], t_exa_c[i][j], f"Exact_c", "green")


        print("\nEuler Method: Explicit, (Temperature Ambient CONSTANT)")
        # Using the explicit euler for each k
        for j, k in enumerate(k_n):
            cont_start = time.perf_counter()
            T_exp_c[i][j], t_exp_c[i][j] = explicit_c(cooling_ode, T0, k, T_amb, t_start, t_end, n_steps)
            cont_end = time.perf_counter()

            cont_exp_c[i][j] = cont_end - cont_start # Time for each solution (k * h)

            plot_temperature(T_exp_c[i][j], t_exp_c[i][j], f"Explicit_Euler_method_c", "blue")
            err_exp_c[i][j] = global_error(T_exa_c[i][j], T_exp_c[i][j])
            print(f"Error[{j}] = {err_exp_c[j]} ")
            print(f"Time taken: {cont_exp_c[i][j]} seconds")
            print("\n")

            
        print("Euler Method: Implicit, (Temperature Ambient CONSTANT)")
        # Using the explicit euler for each k
        for j, k in enumerate(k_n):
            cont_start = time.perf_counter()
            T_imp_c[i][j], t_imp_c[i][j] = implicit_c(T0, k, T_amb, t_start, t_end, n_steps)
            cont_end = time.perf_counter()

            cont_imp_c[i][j] = cont_end - cont_start # Time for each solution (k * h)

            plot_temperature(T_imp_c[i][j], t_imp_c[i][j], f"Implicit_Euler_method_c", "red")
            err_imp_c[i][j] = global_error(T_exa_c[i][j], T_imp_c[i][j])
            print(f"Error[{j}] = {err_imp_c[j]} ")
            print(f"Time taken: {cont_imp_c[i][j]} seconds")
            print("\n")



        save_plots(f"all_methods(k={k:.2f}|h={h:.2f})_c", f"all_methods_k{k:.2f}_h{h:.2f}_c")
        print(f"\n\n")

    # Plot error for each k
    plot_convergence(h_list, err_exp_c, err_imp_c, num_h, k_n, 'c')



    for i, h in enumerate(h_list):
        print(f"------------- h = {h} -------------")
        n_steps = int((t_end - t_start) / h) # Number of parts to be divided

        print("Exact Solution, (Temperature Ambient Variable)")
        # Using the explicit euler for each k
        for j, k in enumerate(k_n):
            cont_start = time.perf_counter()
            T_exa_v[i][j], t_exa_v[i][j] = exact_v(T0, k, t_start, t_end, n_steps)
            cont_end = time.perf_counter()

            cont_exa_v[i][j] = cont_end - cont_start # Time for each solution (k * h)
            print(f"Time taken: {cont_exa_v[i][j]} seconds")

            print("\n")
            plot_temperature(T_exa_v[i][j], t_exa_v[i][j], f"Exact_v", "green")


        print("\nEuler Method: Explicit, (Temperature Ambient Variable)")
        # Using the explicit euler for each k
        for j, k in enumerate(k_n):
            cont_start = time.perf_counter()
            T_exp_v[i][j], t_exp_v[i][j] = explicit_v(T0, k, T_amb_func, t_start, t_end, n_steps)
            cont_end = time.perf_counter()

            cont_exp_v[i][j] = cont_end - cont_start # Time for each solution (k * h)

            plot_temperature(T_exp_v[i][j], t_exp_v[i][j], f"Explicit_Euler_method_v", "blue")
            err_exp_v[i][j] = global_error(T_exa_v[i][j], T_exp_v[i][j])
            print(f"Error[{j}] = {err_exp_v[j]} ")
            print(f"Time taken: {cont_exp_v[i][j]} seconds")
            print("\n")

            
        print("Euler Method: Implicit, (Temperature Ambient Variable)")
        # Using the explicit euler for each k
        for j, k in enumerate(k_n):
            cont_start = time.perf_counter()
            T_imp_v[i][j], t_imp_v[i][j] = implicit_v(T0, k, T_amb_func, t_start, t_end, n_steps)
            cont_end = time.perf_counter()

            cont_imp_v[i][j] = cont_end - cont_start # Time for each solution (k * h)

            plot_temperature(T_imp_v[i][j], t_imp_v[i][j], f"Implicit_Euler_method_v", "red")
            err_imp_v[i][j] = global_error(T_exa_v[i][j], T_imp_v[i][j])
            print(f"Error[{j}] = {err_imp_v[j]} ")
            print(f"Time taken: {cont_imp_v[i][j]} seconds")
            print("\n")



        save_plots(f"all_methods(k={k:.2f}|h={h:.2f})_v", f"all_methods_k{k:.2f}_h{h:.2f}_v")
        print(f"\n\n")

    # Plot error for each k
    plot_convergence(h_list, err_exp_v, err_imp_v, num_h, k_n, 'v')



if __name__ == "__main__":
    main()
