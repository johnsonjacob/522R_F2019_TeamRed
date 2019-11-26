from PID import PID

class SteeringPid():
    '''
    takes care of the pid controllers.
    calling run_pid with the offset and angle seen by the car
    should run all the needed functions to implement a pid control on the car.
    '''
    def __init__(self, expected_offset, kp=0.1, ki=0.010, kd=1.5):
        self._offset_pid = PID(kp, ki, kd)
        self._offset_pid.set_expected(expected_offset)
        self._offset_pid.set_i_limits(-1000,1000)
        self._offset_pid.set_min_max_in(0,300)

    def run_pid(self, offset):
        offset_out = self._offset_pid.run_pid(offset)
        return offset_out
