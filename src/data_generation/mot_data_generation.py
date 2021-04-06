import numpy as np

import itertools


class Object:

    def __init__(self, pos, vel, t, delta_t, sigma, id):
        self.pos = pos
        self.vel = vel
        self.delta_t = delta_t
        self.sigma = sigma
        self.state_history = np.array([np.concatenate([pos,vel,np.array([t])])])
        self.process_noise_matrix = sigma*np.array([[delta_t ** 3 / 3, delta_t ** 2 / 2], [delta_t ** 2 / 2, delta_t]])

        # Unique identifier for every object
        self.id = id

    def update(self, t, rng):
        """
        Updates this object's state using a discretized constant velocity model.
        """

        # Update position and velocity of the object in each dimension separately
        assert len(self.pos) == len(self.vel)
        process_noise = rng.multivariate_normal([0, 0], self.process_noise_matrix, size=len(self.pos))
        self.pos += self.delta_t * self.vel + process_noise[:,0]
        self.vel += process_noise[:,1]

        # Add current state to previous states
        self.state_history = np.vstack((self.state_history,np.concatenate([self.pos.copy(),self.vel.copy(),np.array([t])])))

    def __repr__(self):
        return 'id: {}, pos: {}, vel: {}'.format(self.id, self.pos, self.vel)


class MotDataGenerator:
    def __init__(self, args, rng):
        self.start_pos_params = [args.data_generation.mu_x0, args.data_generation.std_x0]
        self.start_vel_params = [args.data_generation.mu_v0, args.data_generation.std_v0]
        self.prob_add_obj = args.data_generation.p_add
        self.prob_remove_obj = args.data_generation.p_remove
        self.delta_t = args.data_generation.dt
        self.process_noise_intens = args.data_generation.sigma_q
        self.prob_measure = args.data_generation.p_meas
        self.measure_noise_intens = args.data_generation.sigma_y
        self.n_average_false_measurements = args.data_generation.n_avg_false_measurements
        self.n_average_starting_objects = args.data_generation.n_avg_starting_objects
        self.field_of_view_lb = args.data_generation.field_of_view_lb
        self.field_of_view_ub = args.data_generation.field_of_view_ub
        self.max_objects = args.data_generation.max_objects
        self.rng = rng
        self.dim = len(self.start_pos_params[0])

        self.debug = False
        assert self.n_average_starting_objects != 0, 'Datagen does not currently work with n_avg_starting_objects equal to zero.'

        self.t = None
        self.objects = None
        self.trajectories = None
        self.measurements = None
        self.unique_ids = None
        self.unique_id_counter = None
        self.reset()

    def reset(self):
        self.t = 0
        self.objects = []
        self.trajectories = {}
        self.measurements = np.array([])
        self.unique_ids = np.array([], dtype='int64')
        self.unique_id_counter = itertools.count()

        # Add initial set of objects (re-sample until we get a nonzero value)
        n_starting_objects = 0
        while n_starting_objects == 0:
            n_starting_objects = self.rng.poisson(self.n_average_starting_objects)
        self.add_objects(n_starting_objects)

        # Measure the initial set of objects
        self.generate_measurements()

        if self.debug:
            print(n_starting_objects, 'starting objects')

    def create_new_object(self, pos, vel):
        return Object(pos=pos,
                      vel=vel,
                      t=self.t,
                      delta_t=self.delta_t,
                      sigma=self.process_noise_intens,
                      id=next(self.unique_id_counter))

    def add_objects(self, n):
        """
        Adds `n` new objects to `objects` list.
        """
        # Never add more objects than the maximum number of allowed objects
        n = min(n, self.max_objects-len(self.objects))
        if n == 0:
            return

        # Create new objects and save them in the datagen
        positions = self.rng.uniform(low=self.field_of_view_lb, high=self.field_of_view_ub, size=(n,self.dim))
        velocities = self.rng.multivariate_normal(self.start_vel_params[0], self.start_vel_params[1], size=(n,))
        self.objects += [self.create_new_object(pos, vel) for pos,vel in zip(positions, velocities)]

    def remove_far_away_objects(self):
        if len(self.objects) == 0:
            return

        positions = np.array([obj.pos for obj in self.objects])
        lb = positions < self.field_of_view_lb
        ub = positions > self.field_of_view_ub
        remove_elements = np.bitwise_or(lb.any(axis=1), ub.any(axis=1))

        self.objects = [o for o, r in zip(self.objects, remove_elements) if not r]

    def remove_objects(self, p):
        """
        Removes each of the objects with probability `p`.
        """

        # Compute which objects are removed in this time-step
        deaths = self.rng.binomial(n=1, p=p, size=len(self.objects))

        n_deaths = sum(deaths)
        if self.debug and (n_deaths > 0):
            print(n_deaths, 'objects were removed')

        # Save the trajectories of the removed objects
        for obj, death in zip(self.objects, deaths):
            if death:
                self.trajectories[obj.id] = obj.state_history

        # Remove them from the object list
        self.objects = [o for o, d in zip(self.objects, deaths) if not d]

    def get_prob_death(self, obj):
        return self.prob_remove_obj

    def remove_object(self, obj, p = None):
        """
        Removes an object based on its state
        """
        if p is None:
            p = self.get_prob_death(obj)

        r = self.rng.rand()

        if r < p:
            return True
        else:
            return False

    def generate_measurements(self):
        """
        Generates all measurements (true and false) for the current time-step.
        """
        # Generate the measurement for each object with probability `self.prob_measure`
        is_measured = self.rng.binomial(n=1, p=self.prob_measure, size=len(self.objects))
        measured_objects = [obj for obj, is_measured in zip(self.objects, is_measured) if is_measured]
        measurement_noise = self.rng.normal(0, self.measure_noise_intens, size=(len(measured_objects),self.dim))
        true_measurements = np.array([np.append(obj.pos+noise, self.t) for obj, noise in zip(measured_objects, measurement_noise)])

        # Generate false measurements
        n_false_measurements = self.rng.poisson(self.n_average_false_measurements)
        false_meas = self.rng.uniform(self.field_of_view_lb, self.field_of_view_ub, size=(n_false_measurements,self.dim))
        false_measurements = np.ones((n_false_measurements,self.dim+1)) * self.t
        false_measurements[:,:-1] = false_meas

        # Also save from which object each measurement came from (for contrastive learning later); -1 is for false meas.
        unique_obj_ids_true = [obj.id for obj in measured_objects]
        unique_obj_ids_false = [-1]*len(false_measurements)
        unique_obj_ids = np.array(unique_obj_ids_true + unique_obj_ids_false)

        # Concatenate true and false measurements in a single array
        if true_measurements.shape[0] and false_measurements.shape[0]:
            new_measurements = np.vstack([true_measurements, false_measurements])
        elif true_measurements.shape[0]:
            new_measurements = true_measurements
        elif false_measurements.shape[0]:
            new_measurements = false_measurements
        else:
            return

        # Shuffle all generated measurements and corresponding unique ids in unison
        random_idxs = self.rng.permutation(len(new_measurements))
        new_measurements = new_measurements[random_idxs]
        unique_obj_ids = unique_obj_ids[random_idxs]

        # Save measurements and unique ids
        self.measurements = np.vstack([self.measurements, new_measurements]) if self.measurements.shape[0] else new_measurements
        self.unique_ids = np.hstack([self.unique_ids, unique_obj_ids])

    def step(self, add_new_objects=True):
        """
        Performs one step of the simulation.
        """
        self.t += self.delta_t

        # Update the remaining ones
        for obj in self.objects:
            obj.update(self.t, self.rng)

        # Remove objects that left the field-of-view
        self.remove_far_away_objects()

        # Add new objects
        if add_new_objects:
            n_new_objs = self.rng.poisson(self.prob_add_obj)
            self.add_objects(n_new_objs)

        # Remove some of the objects
        self.remove_objects(self.prob_remove_obj)
        
        # Generate measurements
        self.generate_measurements()
        
        if self.debug:
            if n_new_objs > 0:
                print(n_new_objs, 'objects were added')
            print(len(self.objects))

    def finish(self):
        """
        Should be called after the last call to `self.step()`. Removes the remaining objects, consequently adding the
        remaining parts of their trajectories to `self.trajectories`.
        """
        self.remove_objects(1.0)
