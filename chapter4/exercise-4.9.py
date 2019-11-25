import numpy as np
import matplotlib.pyplot as plt

# initializing value of the states
V_s = np.zeros(101)
new_V_s = np.zeros(101)
V_s[100] = 1
new_V_s[100] = 1
pi = np.zeros(100)

# hyperparameters
theta = 1.0e-9
p_h = 0.25

# policy iteration
while True:
	max_error = 0
	V_s = np.copy(new_V_s)
	for s in range(1, 100): # 1,2,3 ... 99
		v = V_s[s]
		max_action_value = float("-inf")
		for a in range(0, min(s + 1, 101 - s)): # 0,1,2, .... min(s,100-s)
			if s + a >= 100:
				r = 1
			else:
				r = 0
				
			win_transition_return = p_h * (r + V_s[s + a])

			r = 0 # reseting for bad transition return
			loss_transition_return = (1 - p_h) * (r + V_s[s - a])
			action_value = win_transition_return + loss_transition_return
			max_action_value = max(action_value, max_action_value)
		
		new_V_s[s] = max_action_value

		max_error = max(abs(v - new_V_s[s]), max_error)

	if max_error < theta:
		V_s = np.copy(new_V_s)
		break

# outputing a deterministic policy 
for s in range(1, 100): # 1,2,3 ... 99
	optimal_a = -1
	max_action_value = float("-inf")
	for a in range(0, min(s + 1, 101 - s)): # 0,1,2, .... min(s,100-s)
		if s + a >= 100:
			r = 1
		else:
			r = 0
		
		win_transition_return = p_h * (r + V_s[s + a])

		r = 0 # reseting for bad transition return
		loss_transition_return = (1 - p_h) * (r + V_s[s - a])
		action_value = win_transition_return + loss_transition_return
		if action_value > max_action_value and a != 0: # ignoring the action 0
			if abs(action_value - max_action_value) > 0.000001: # only choosing the action if it provides a decent benefit
				max_action_value = action_value
				optimal_a = a
				pi[s] = optimal_a

indices = [x for x in range(99)]
plt.scatter(indices, V_s[0:99])
plt.show()

plt.scatter(indices, pi[1:100])
plt.show()