__description__ = \
"""
Class for running Conway's Game of Life.
"""
__author__ = "Michael J. Harms"
__date__ = "2016-07-15"

import numpy as np

class Conway:
    """
    Class for playing Conway's Game of Life.
    """

    def __init__(self,x_size=28,y_size=28,starting_density=None):
        """
        Initialize the grid to zeros.  If a starting density is specified,
        create a random initial state.
        """

        self._x_size = x_size
        self._y_size = y_size
        self._starting_density = starting_density

        self._current_state = np.zeros((self._x_size,self._y_size),dtype=bool)
        self._next_state = np.zeros((self._x_size,self._y_size),dtype=bool)

        if self._starting_density != None:
            self.random_state(self._starting_density)

    def random_state(self,starting_density=0.5):
        """
        Generate a random game state.
        """
   
        self._starting_density = starting_density
        self._current_state = np.random.rand(self._x_size,self._y_size) > self._starting_density

    def iterate(self):
        """
        Calculate the next iteration of the game.
        """

        for i in range(self._x_size):
            for j in range(self._y_size):
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
        if i + 1 == self._x_size:
            l1[2] = -1
        elif i + 1 > self._x_size:
            err = "Bound error."
            raise ValueError(err)
        else:
            pass
       
        # detremine neighbor j indexes 
        l2 = [j-1,j,j+1]
        if j + 1 == self._y_size:
            l2[2] = -1
        elif j + 1 > self._y_size:
            err = "Bound error."
            raise ValueError(err)
        else:
            pass

        sub = self._current_state[[[l1[0]],[l1[1]],[l1[2]]],l2]

        return np.sum(sub) - sub[1,1]

    @property
    def state(self):
        """
        Return the state of the game.
        """
           
        return self._current_state 

    @property
    def dimensions(self):
        """
        Return grid dimensions.
        """

        return (self._x_size,self._y_size)
