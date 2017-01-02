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

import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt

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
            self._initial_state = np.zeros((self._y_size,self._x_size),dtype=np.bool)
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
        self.reset()

    def reset(self):

        # Reset the iterator
        self._current_state = np.copy(self._initial_state)
        
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

        # Construct initial state matrix
        self._x_size = len(list(lines[0].strip()))
        self._y_size = len(lines) 
        self._initial_state = np.zeros((self._y_size,self._x_size),dtype=np.bool)
    
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

            self._initial_state[i,:] = row

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
        self._initial_state = np.random.rand(self._y_size,self._x_size) < self._starting_density


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

    def as_rgba(self,cmap=cm.copper,history_length=100,flip=False):
        """
        Return the current Conway state as RGBA, using the cmap specified. The
        cmap should be an instance of 
        """
        
        # Put each pixel on a 0-1 scale capturing its state ranging from 
        # alive to previously alive to dead
        to_plot = np.zeros(self.current_state.shape,dtype=np.float)
        to_plot[self.current_state] = 1.0
        step = 1.0/(history_length + 2)
        for i in range(history_length):
            to_plot[self.time_since_alive == i] = 1 - (i + 1)*step
        
        # Flip color scale if requested
        if flip:
            to_plot = 1 - to_plot

        # Put on 0-254 scale
        to_plot = to_plot*254

        plt.imshow(to_plot,cmap=cmap,interpolation="nearest")
        plt.show()
        
        # Return RGBA color map
        return cmap(to_plot)
        
