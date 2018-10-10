import time

for i in range(5):
    time.sleep(1)
    print('\r{}'.format("#########"), end='')
print('\r{}'.format(""), end='')
print("Finished", end='')
