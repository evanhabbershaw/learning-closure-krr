"""NumPy Lorenz-63 reference; generation is explicit."""
import numpy as np
train_seed = 22
test_seed = 43

def L63(u, alpha=10., rho=28., beta=8./3.):

    x, y, z = u 
    p = alpha * (y - x)
    q = (rho - z) * x - y
    r = x * y - beta * z
    return np.array([p, q, r])


def rk4_step(f, u, dt):
    k1 = f(u)
    k2 = f(u + 0.5*dt*k1)
    k3 = f(u + 0.5*dt*k2)
    k4 = f(u + dt*k3)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def generate_trajectory(state0, dt, n_steps):
    traj = np.zeros((n_steps, 3), dtype=float)
    u = np.array(state0, dtype=float)
    traj[0] = u
    for n in range(1, n_steps):
        u = rk4_step(L63, u, dt)
        traj[n] = u
    return traj


def gen_data(dt=0.01, train_size=int(1e5), test_num=500, test_size=2500):

    np.random.seed(train_seed)
    g0 =  np.random.normal(size=(3))
    state0 = generate_trajectory(g0, dt, int(40/dt))[-1]
    train = generate_trajectory(state0, dt, train_size)
    np.random.seed(test_seed)

    g0 =  np.random.normal(size=(3))
    state0 = generate_trajectory(g0, dt, int(40/dt))[-1]
    test = generate_trajectory(state0, dt, test_num*test_size)
    test = np.moveaxis(test.reshape(test_num, -1, 3), 1, 2)
    np.random.shuffle(test)

    return train, test
