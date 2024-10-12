import random
import datetime
import time

trial_amount = 1000
WEIGHTS = {'accuracy': 0.6, 'time': 0.3, 'cost': 0.1}

# initial cases
cases1 = [
    ['large'],
    ['large'] * 5,
    ['medium', 'small'],
    ['medium', 'medium', 'small'],
    ['small', 'medium', 'large']
]

# all possible iterations of a 3 LLM system
cases2 = [
    ['large', 'large', 'large'],
    ['large', 'large', 'medium'],
    ['large', 'large', 'small'],
    ['large', 'medium', 'large'],
    ['large', 'medium', 'medium'],
    ['large', 'medium', 'small'],
    ['large', 'small', 'large'],
    ['large', 'small', 'medium'],
    ['large', 'small', 'small'],
    ['medium', 'large', 'large'],
    ['medium', 'large', 'medium'],
    ['medium', 'large', 'small'],
    ['medium', 'medium', 'large'],
    ['medium', 'medium', 'medium'],
    ['medium', 'medium', 'small'],
    ['medium', 'small', 'large'],
    ['medium', 'small', 'medium'],
    ['medium', 'small', 'small'],
    ['small', 'large', 'large'],
    ['small', 'large', 'medium'],
    ['small', 'large', 'small'],
    ['small', 'medium', 'large'],
    ['small', 'medium', 'medium'],
    ['small', 'medium', 'small'],
    ['small', 'small', 'large'],
    ['small', 'small', 'medium'],
    ['small', 'small', 'small'],
]

# define attributes of each LLM
LLM_PARAMS = {
    'large': {'accuracy': 0.8, 'cost': 40.5, 'cold_start': 600, 'exec_time': 1},
    'medium': {'accuracy': 0.7, 'cost': 7, 'cold_start': 180, 'exec_time': 0.5},
    'small': {'accuracy': 0.6, 'cost': 0.8, 'cold_start': 30, 'exec_time': 0.2}
}

# used for normalizing time and cost
max_time = 1  # (case of 5 calls to large LL all fail)
max_cost = 375   # (case of 5 calls to large LLM)

# normalization function
def normalize(value, max_value):
    return value / max_value

# mean function (for finding average across all cases)
def calculate_mean(values):
    return sum(values) / len(values) if values else 0

class EventSimulator:
    def __init__(self, event_list=None):
        self.event_list = event_list if event_list is not None else []
        self.event_list.sort()

    def add_event(self, e):
        self.event_list.append(e)
        self.event_list.sort()

    def step(self):
        if not self.event_list:
            print("No events to process.")
            return
        e = self.event_list.pop(0)
        generated_events = e.do()
        for i in generated_events:
            self.add_event(i)

class GenericEvent:
    def __init__(self, name, time):
        self.name = name
        self.time = time

    def __lt__(self, other):
        return self.time < other.time
    
    def _effect(self):
        print("At {}: {}".format(self.time.isoformat(), self.name))

    def do(self):
        raise NotImplementedError

class LLMEvent(GenericEvent):
    def __init__(self, name, llm_type, time):
        super().__init__(name, time)
        self.llm_type = llm_type
        self.execution_time = LLM_PARAMS[llm_type]['exec_time']
        self.memory_cost = LLM_PARAMS[llm_type]['cost']
        self.accuracy = LLM_PARAMS[llm_type]['accuracy']

    def do(self):
        self._effect()
        print(f"Executing {self.name} ({self.llm_type}) with execution time: {self.execution_time:.2f}s")
        # time.sleep(self.execution_time / 5) (uncomment/comment to simulate time delay)
        return []

class VerifierEvent(GenericEvent):
    def __init__(self, llm_events, time):
        super().__init__("Verifier Event", time)
        self.llm_events = llm_events
        self.detected_accuracy = 0  # accuracy changes if an output is verified
        self.verification_count = 0 # each verification takes time (must keep track of iterations)
        self.max_generation_time = 0 # simulates parallel aspect of LLM generation

    def do(self):
        print("Verifying outputs from LLMs...")
        for llm_event in self.llm_events:
            # update time and count each iteration
            self.max_generation_time = max(self.max_generation_time, llm_event.execution_time)
            self.verification_count += 1
            
            # simulate check if the output is correct based on random value and accuracy of LLM
            if random.random() < llm_event.accuracy:
                self.detected_accuracy = 0.9 # update accuracy if correct output detected
                print(f"Correct output detected from {llm_event.name} ({llm_event.llm_type}). Setting system accuracy to {self.detected_accuracy:.2f}")
                break
        # if no correct output was detected, pick a random one
        if self.detected_accuracy == 0:
            random_llm_event = random.choice(self.llm_events)
            self.detected_accuracy = random_llm_event.accuracy
            print(f"No correct output detected. Returning a random output from {random_llm_event.name}.")

        return [] 

# run simulation for each case
def run_simulation(case):
    sim = EventSimulator()
    current_time = datetime.datetime.now()

    llm_events = []
    total_time = 0
    total_cost = 0

    for llm_type in case:
        start_time = current_time + datetime.timedelta(seconds=LLM_PARAMS[llm_type]['cold_start'])
        llm_event = LLMEvent(f"LLM-{llm_type}", llm_type, start_time)
        llm_events.append(llm_event)
        sim.add_event(llm_event)

    while sim.event_list:
        sim.step()

    verifier_time = current_time + datetime.timedelta(seconds=1)
    verifier_event = VerifierEvent(llm_events, verifier_time)
    sim.add_event(verifier_event)
    sim.step()

    # calculate total time (LLM output generation + verify iterations)
    total_time = verifier_event.max_generation_time + (0.1 * verifier_event.verification_count)

    # calculate total cost (sum)
    for llm_event in llm_events:
        total_cost += llm_event.memory_cost

    # accuracy is an attribute of the verifier
    accuracy = verifier_event.detected_accuracy

    return accuracy, total_time, total_cost

def evaluate_cases(cases):
    results = []
    abbreviation_map = {
        'large': 'L',
        'medium': 'M',
        'small': 'S'
    }
    
    for i, case in enumerate(cases, 1):
        accuracies, times, costs = [], [], []
        for _ in range(trial_amount):
            accuracy, time, cost = run_simulation(case)
            accuracies.append(accuracy)
            times.append(time)
            costs.append(cost)
        
        avg_acc = calculate_mean(accuracies)
        avg_time = calculate_mean(times)
        avg_cost = calculate_mean(costs)
        objective = (WEIGHTS['accuracy'] * avg_acc - WEIGHTS['time'] * normalize(avg_time, max_time) - WEIGHTS['cost'] * normalize(avg_cost, max_cost))

        print(f"Case {i} Results:")
        print(f"  Accuracy: {avg_acc:.3f}")
        print(f"  Time: {normalize(avg_time, max_time):.3f}")
        print(f"  Cost: {normalize(avg_cost, max_cost):.3f}")
        print(f"  Objective: {objective:.3f}\n")

        # save case, case abbreviation and its objective values at the end
        case_abbrev = ''.join([abbreviation_map[llm_type] for llm_type in case])
        results.append({
            "case_number": i,
            "case_abbrev": case_abbrev,
            "objective": objective
        })

    # print all cases and their objective values after all cases have run
    print("All cases objective values are printed below:\n")
    for result in results:
        print(f"Case {result['case_number']}: {result['case_abbrev']} - Objective value: {result['objective']:.3f}\n")
    # return optimal solution sa the max objective value
    optimized_case = max(results, key=lambda x: x['objective'])
    print("Optimized Case:")
    print(f"Case {optimized_case['case_number']}: {optimized_case['case_abbrev']} - Objective value: {optimized_case['objective']:.3f}")

# uncomment and comment to switch the cases tested

evaluate_cases(cases1)
# evaluate_cases(cases2)
