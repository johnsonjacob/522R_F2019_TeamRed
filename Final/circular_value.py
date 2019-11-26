

class Circular_Value:

    def __init__(self, value=4):

        self.value = value - 1

    def next(self, current):
        if current > self.value:
            raise Exception("Current: {} is Larger than Value: {}".format(current, self.value))

        if current == self.value:
            return 0
        return current + 1

    def previous(self, current):
        if current > self.value:
            raise Exception("Current: {} is Larger than Value: {}".format(current, self.value))

        if current == 0:
            return self.value 
        return current - 1

if __name__ == "__main__":
    cv = Circular_Value()
    for i in range(5):
        try:
            print("current: {}, next: {}".format(i, cv.next(i)))
        except Exception as e:
            print(repr(e))
    for i in range(5):
        try:
            print("current: {}, previous: {}".format(i, cv.previous(i)))
        except Exception as e:
            print(repr(e))
