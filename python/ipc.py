import multiprocessing
import time

'''

Inter Process Communication using queue concept.

Demonstraightion of 
    - communication between multiple producers and a consumer
    - optimum use of time
    - graceful exit

'''

class Producer(multiprocessing.Process):
    def __init__(self, task_q, sleep_time):
        multiprocessing.Process.__init__(self)
        self.results_q = multiprocessing.Queue()
        self.task_q = task_q
        self.__exit_counter = 0
        self.__consumed = True
        self.__EXIT_COUNT = 50

    def run(self):
        proc_name = self.name
        task = {}

        while True:
            try:
                if self.__exit_counter == self.__EXIT_COUNT:
                    break

                if not self.task_q.full() and self.__consumed:
                    task[self.name] = self.__exit_counter
                    self.task_q.put(task)
                    print(self.name, "produced = ", self.__exit_counter)
                    self.__consumed = False

                time.sleep(sleep_time)

                if not self.results_q.empty():
                    results = self.results_q.get()
                    print(self.name, "consummed = ", results)
                    self.__consumed = True

                self.__exit_counter += 1
            except:
                pass

        print(self.name, "exiting")
        del self.results_q
        return

    def stop(self):
        self.__exit_counter = self.__EXIT_COUNT


if __name__ == '__main__':
    sleep_time = 0.1
    num_producers = 2

    tasks_q = multiprocessing.Queue(maxsize=100)

    producers = [ Producer(tasks_q, sleep_time) for i in range(num_producers) ]
    producers_dict = {}

    for producer in producers:
        producers_dict[producer.name] = producer

    print(producers_dict)

    for w in producers:
        w.start()

    exit_counter = 0

    while True:
        try:
            if not tasks_q.empty():
                task = tasks_q.get()
                (key, value) = task.popitem()
                producers_dict[key].results_q.put(value)
                exit_counter = 0

            exit_counter += 1
            if exit_counter == 100:
                break
            time.sleep(sleep_time/num_producers)
        except:
            print("\napp exit requested")
            break

    del tasks_q

    for w in producers:
        w.stop()
        w.join()

    print("app exit")