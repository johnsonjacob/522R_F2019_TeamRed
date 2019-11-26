class PID():
    '''
    implements a generic pid controller

    the methods of use are set_expected and run_pid
    run_pid needs the current state of the car passed in to calculate the error.
    if kp ki and kd need to be updated after instatiation (I don't think they should...) feel free to reach in
    '''

    def __init__(self, kp, ki, kd):
        '''setup the kp = porportional, ki = integral, and kd = derivative constants'''
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._expected = 0
        self._current = 0
        self._previous_error = 0
        self._sum_error = 0
        self._max_i = None
        self._min_i = None
        self._min_in = None
        self._max_in = None
        self._min_mult = 1
        self._max_mult = 1
    
    def set_expected(self, value):
        '''
        set expected state value for the pid controller
        with an angle this might in general be 0
        with an offset it would be the value for the correct offset
        '''
        self._expected = value
        self._real_expected = value

    def run_pid(self, current_state):
        '''
        run the pid calculations
        update the controller with the current state
        '''
        #print("in " + str(current_state))
        #the following if statement centers the values
        if self._min_in != None and self._max_in != None:
            if current_state < self._real_expected:
                current_state = current_state - self._min_in
                current_state = self._min_mult * current_state
                current_state = current_state + self._min_in
            else:
                current_state = current_state - self._real_expected
                current_state = current_state * self._max_mult
                current_state = current_state + self._expected
        #print("out " + str(current_state))
        self._set_current(current_state)
        return self._run_pid()
        
    def set_i_limits(self, min_value, max_value):
        '''
        give the controller a maximum and minimum value for the i
        set to None if you don't want to have a max or minimum
        '''
        self._max_i = max_value
        self._min_i = min_value

    def set_min_max_in(self, min_value, max_value):
        self._min_in = min_value
        self._max_in = max_value
        self._real_expected = self._expected
        self._expected = (max_value + min_value) / 2
        self._min_mult = self._expected/self._real_expected
        self._max_mult = (max_value - self._expected)/(max_value - self._real_expected)

    def _set_current(self, value):
        '''
        set current value for the pid controller
        this is where our device currently is
        '''
        self._current = value

    def _run_pid(self):
        '''
        calculate the output value for the current pid calculation
        '''
        error = self._current - self._expected 
        p_part = self._get_p(error)
        i_part = self._get_i(error)
        d_part = self._get_d(error)
        result =  p_part + i_part + d_part
        if p_part < 0:
            if result >0:
                result = min(0, result)
        if p_part > 0:
            if result < 0:
                result = max(0, result)
        return result

    def _get_p(self, error):
        '''
        this will get the porportional term for the calculation
        '''
        #print("porportional term of the errors: " + str(error))
        return error * self.kp

    def _get_i(self, error):
        '''
        this will get the integral term for the calculation
        '''
        self._sum_error += error
        if(self._max_i != None):
            if self._sum_error > self._max_i:
                self._sum_error = self._max_i
        if(self._min_i != None):
            if self._sum_error < self._min_i:
                self._sum_error = self._min_i
        #print("integral term of the errors: "  + str(self._sum_error))
        return self._sum_error * self.ki

    def _get_d(self, error):
        '''
        this will get the derivative term for the calculation
        '''
        current_d = error - self._previous_error
        self._previous_error = error
        #print("derivative term of the errors: " + str(current_d))
        return current_d * self.kd

    
    

