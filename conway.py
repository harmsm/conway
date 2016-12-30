__description__ = \
"""
Class for running Conway's Game of Life.
"""
__author__ = "Michael J. Harms"
__date__ = "2016-07-15"

import time
import numpy as np
np.set_printoptions(threshold=np.nan)
import scipy.signal

class Conway:
    """
    Class for playing Conway's Game of Life.
    """

    def __init__(self,x_size=28,y_size=28,starting_density=0.5,seed=None,
                 state_file=None):
        """
        Initialize the grid to zeros.  If a starting density is specified,
        create a random initial state.  If a configuration file is specified, 
        initialize the grid with its contents. The configuration file will 
        override all other arguments specified.
        """

        # Parse arguments
        self._x_size = x_size
        self._y_size = y_size
        self._starting_density = starting_density
        self._seed = seed
        self._state_file = state_file

        # Either read a configuration file or generate a random starting state
        if self._state_file:
            self._read_state_file(state_file)           
        else: 
            self._current_state = np.zeros((self._x_size,self._y_size),dtype=np.bool)
            if not self._seed:
                self._seed = np.random.randint(0,high=2**32)
            self._random_state(self._starting_density)

        # Create kernel for convolution/neighbor counting
        # 1 1 1
        # 1 0 1
        # 1 1 1
        self._donut_kernel = np.ones((3,3),dtype=int)
        self._donut_kernel[1,1] = 0

        # Initial state of the system
        self._initial_state = np.copy(self._current_state)

        # Record number of time steps that each cell is alive over sim
        self._time_alive = np.zeros(self._current_state.shape,dtype=np.int)
        self._time_alive += self._current_state

        # Construct array of -1 (never alive) and 0 (currently alive).  At each 
        # time step, time_since_alive will increase by one for all cells that
        # were alive at some point but currently dead. When a dead cell becomes
        # alive, the counter resets to 0 for that cell. 
        self._time_since_alive = np.ones(self._current_state.shape,dtype=np.int)*-1
        self._time_since_alive += self._current_state

    def _read_state_file(self,state_file):
        """
        Read a file with a starting configuration.  This file should have the 
        format:

        0100101... 
        1010101...
        ........
        ....... .
        .......  .

        where the state of each cell is defined as 0 or 1.  All rows must have
        the same number of columns.  Blank lines and lines starting with "#"
        are ignored.
        """

        # Read file
        f = open(state_file,'r')
        lines = f.readlines()
        f.close()
    
        # Skip comments and blank lines
        lines = [l for l in lines if not l.startswith("#") and
                 len(l.strip()) != 0]

        # Construct current state matrix
        self._x_size = len(list(lines[0].strip()))
        self._y_size = len(lines) 
        self._current_state = np.zeros((self._y_size,self._x_size),dtype=np.bool)
    
        # Read the file.
        for i, l in enumerate(lines):
            try:
                row = [bool(int(c)) for c in list(l.strip())]
            except ValueError:
                err = "Row {:} has a non-boolean value somewhere.\n".format(i)
                raise ValueError(err) 
            if len(row) != self._x_size:
                err = "Row {:} does not have {:} columns.\n".format(i,self._x_size)
                raise ValueError(err)

            self._current_state[i,:] = row

    def write_state_file(self,out_file,initial_state=True):
        """
        Write a configuration file that can be read by the _read_state_file 
        method.
        """

        if initial_state:
            arr = self._initial_state
        else:
            arr = self._current_state

        out = np.array2string(arr,
                              separator='',
                              formatter={'bool':lambda x:"{:d}".format(x)})
        out = out.replace("[","")
        out = out.replace("]","")
        lines = out.split("\n")
        out = "\n".join([l.strip() for l in lines])
      
        f = open(out_file,'w')
        if initial_state:
            to_write = "Initial"
        else:
            to_write = "Current (iteration {})"

        f.write("# Conway Game of Life State\n")
        f.write("# State: {} \n".format(to_write))
        f.write("# Time: {}\n".format(time.asctime()))
        if self.seed:
            f.write("# Random state. \n")
            f.write("# Seed: {}\n".format(self.seed))
            f.write("# Density: {:.10e}\n".format(self._starting_density))

        f.write(out)
        f.close()
     
    def _random_state(self,starting_density=0.5):
        """
        Generate a random game state.
        """
        
        # Make sure we're using the correct starting seed     
        np.random.seed(self.seed)
   
        self._starting_density = starting_density
        self._current_state = np.random.rand(self._y_size,self._x_size) > self._starting_density


    def iterate(self):

        # Count neighbors of each cell
        num_neighbors = scipy.signal.convolve2d(self._current_state,
                                                self._donut_kernel,
                                                mode="same",
                                                boundary="wrap")

        # Create masks for different conditions
        alone_mask = (self._current_state == 1)*(num_neighbors  < 2)
        crowd_mask = (self._current_state == 1)*(num_neighbors  > 3)
        spawn_mask = (self._current_state == 0)*(num_neighbors == 3)

        # Apply masks
        self._current_state[alone_mask] = 0
        self._current_state[crowd_mask] = 0
        self._current_state[spawn_mask] = 1

        # Record time_alive step for currently living cells
        self._time_alive[self._current_state == 1] += 1

        # The time_since_alive is 0 for all currently living cells
        self._time_since_alive[self._current_state == 1] = 0

        # Create mask for cells that were alive at some point, but are now dead.
        # Increment the time_since_alive for those cells by one.
        time_mask = (self._time_since_alive != -1)*(self._current_state == 0)
        self._time_since_alive[time_mask] += 1

    def iterate_slowly(self):
        """
        Calculate the next iteration of the game.
        """

        self._next_state = np.zeros((self._y_size,self._x_size),dtype=bool)
        for i in range(self._y_size):
            for j in range(self._x_size):
                num_neighbors = self._neighbor_density(i,j)
    
                # living cell
                if self._current_state[i,j] == 1:

                    # die of crowding or loneliness
                    if num_neighbors < 2 or num_neighbors > 3:
                        self._next_state[i,j] = 0
            
                    # happily remain
                    else:
                        self._next_state[i,j] = 1

                # dead cell
                else:
                    # spawn
                    if num_neighbors == 3:
                        self._next_state[i,j] = 1

                    # stay dead
                    else:
                        self._next_state[i,j] = 0
        
      
        self._current_state = np.copy(self._next_state)

 
    def _neighbor_density(self,i,j):
        """
        Calculate the number of neighbors for a given grid point defined by i
        and j. 
        """

        # determine neighbor i indexes
        l1 = [i-1,i,i+1]
        if i + 1 == self._y_size:
            l1[2] = 0 
        elif i + 1 > self._y_size:
            err = "Bound error."
            raise ValueError(err)
        else:
            pass
       
        # detremine neighbor j indexes 
        l2 = [j-1,j,j+1]
        if j + 1 == self._x_size:
            l2[2] = 0 
        elif j + 1 > self._x_size:
            err = "Bound error."
            raise ValueError(err)
        else:
            pass

        sub = self._current_state[[[l1[0]],[l1[1]],[l1[2]]],l2]

        return np.sum(sub) - sub[1,1]

    @property
    def seed(self):
        """
        Seed used to initialize random number generator.
        """
        
        return self._seed

    @property
    def initial_state(self):
        """
        Return initial state of the game.
        """

        return self._initial_state

    @property
    def current_state(self):
        """
        Return the state of the game.
        """
           
        return self._current_state 

    @property
    def dimensions(self):
        """
        Return grid dimensions.
        """

        return self._current_size.shape

    @property
    def time_alive(self):
        """
        Return how often each cell has been alive.
        """

        return self._time_alive

    
    @property
    def time_since_alive(self):
        """
        Return the time since a cell was alive.  If never alive, will be -1.
        If currently alive, will be 0.  If alive last step, but not now, 
        will be 1.  If previously alive, but dead for last three steps, will
        be 3. etc.
        """

        return self._time_since_alive
