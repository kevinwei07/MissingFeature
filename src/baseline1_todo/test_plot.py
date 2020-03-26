from  matplotlib import pyplot as plt
import numpy as np
import json
from ipdb import set_trace as pdb


path = f'missing0+bn+lr+l2/history.json'
with open(path, 'r') as f:
    history = json.loads(f.read())
valid_loss = [(i, loss['loss']) for i, loss in enumerate(history['valid']) if i % 10 == 0]
valid_loss = [*zip(*valid_loss)]
x, y = valid_loss[0], valid_loss[1]
new_ticks = [i for i in range(0, 1500, 100)]

plt.figure()
plt.plot(x, y)
plt.xticks(new_ticks)
plt.savefig('test.png')

exit()

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2

plt.figure()
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')

plt.xlim((-1, 2))
plt.ylim((-2, 3))
plt.xlabel('I am x')
plt.ylabel('I am y')
plt.show()

new_ticks = np.arange(0, 151, 10)
print(new_ticks)
plt.xticks(new_ticks)

plt.yticks([-2, -1.8, -1, 1.22, 3], [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])

plt.savefig('test.png')