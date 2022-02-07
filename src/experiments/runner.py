import os
import subprocess


if __name__ == '__main__':
    experiment_path = 'src/experiments/1_1_compare_constanta_single.py'

    for i in range(50):
        try:
            for epsilon in (.01, .1, .5, 1):
                print(f'Epsilon: {epsilon}, epoch: {i}')
                proc = subprocess.Popen(map(str, ["py", experiment_path, epsilon, 0]),
                                        stdout=subprocess.PIPE,
                                        shell=True)

                out, err = proc.communicate()
                out: bytes = out
                print(out.decode("utf-8"))

                with open('output.txt', 'ab') as f:
                    f.write(out)
                    f.write('\n'.encode())
        except KeyboardInterrupt:
            print('Stopped')
            break
