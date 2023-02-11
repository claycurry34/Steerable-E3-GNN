from synthetic_sim import ChargedParticlesSim, SpringSim, GravitySim
import time
import numpy as np
import argparse

"""
nbody_small:   python3 -u generate_dataset.py --simulation=charged --num-train 10000 --seed 43 --suffix small
gravity_small: python3 -u generate_dataset.py --simulation=gravity --num-train 10000 --seed 43 --suffix small
"""

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='charged',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=10000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=2000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=2000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length_test', type=int, default=5000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n_balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--initial_vel', type=int, default=1,
                    help='consider initial velocity')
parser.add_argument('--suffix', type=str, default="",
                    help='add a suffix to the name')

args = parser.parse_args()

initial_vel_norm = 0.5
if not args.initial_vel:
    initial_vel_norm = 1e-16

if args.simulation == 'springs':
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_springs'
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls, vel_norm=initial_vel_norm)
    suffix = '_charged'
elif args.simulation == 'gravity':
    sim = GravitySim(noise_var=0.0, n_balls=args.n_balls, vel_norm=initial_vel_norm)
    suffix = '_gravity'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls) + "_initvel%d" % args.initial_vel + args.suffix
np.random.seed(args.seed)

print(suffix)


def generate_dataset(num_sims, length, sample_freq, batch_size=1000):
    loc_all = list()
    vel_all = list()
    force_all = list()
    mass_all = list()
    for i in range(0, num_sims, batch_size):
        t = time.time()
        loc, vel, force, mass = sim.sample_trajectory(T=length,
                                                         sample_freq=sample_freq,
                                                         batch_size=batch_size)

        loc_all.append(loc)
        vel_all.append(vel)
        force_all.append(force)
        mass_all.append(mass)

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))

    loc_all = np.vstack(loc_all)
    vel_all = np.vstack(vel_all)
    force_all = np.vstack(force_all)
    mass_all = np.vstack(force_all)

    return loc_all, vel_all, force_all, mass_all


if __name__ == "__main__":

    print("Generating {} training simulations".format(args.num_train))
    loc_train, vel_train, force_train, mass_train = generate_dataset(args.num_train,
                                                                        args.length,
                                                                        args.sample_freq)

    print("Generating {} validation simulations".format(args.num_valid))
    loc_valid, vel_valid, force_valid, mass_valid = generate_dataset(args.num_valid,
                                                                        args.length,
                                                                        args.sample_freq)

    print("Generating {} test simulations".format(args.num_test))
    loc_test, vel_test, force_test, mass_test = generate_dataset(args.num_test,
                                                                    args.length_test,
                                                                    args.sample_freq)

    np.save('loc_train' + suffix + '.npy', loc_train)
    np.save('vel_train' + suffix + '.npy', vel_train)
    np.save('force_train' + suffix + '.npy', force_train)
    np.save('mass_train' + suffix + '.npy', mass_train)

    np.save('loc_valid' + suffix + '.npy', loc_valid)
    np.save('vel_valid' + suffix + '.npy', vel_valid)
    np.save('force_valid' + suffix + '.npy', force_valid)
    np.save('mass_valid' + suffix + '.npy', mass_valid)

    np.save('loc_test' + suffix + '.npy', loc_test)
    np.save('vel_test' + suffix + '.npy', vel_test)
    np.save('force_test' + suffix + '.npy', force_test)
    np.save('mass_test' + suffix + '.npy', mass_test)
