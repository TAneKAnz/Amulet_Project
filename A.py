import os

main_folder = '/Users/tanekanz/CEPP-2/CEPP'

print(os.path.join(os.path.dirname(main_folder), f'Aug{os.path.basename(main_folder)}'))