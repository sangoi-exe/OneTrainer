from util.import_util import script_imports

script_imports()

from modules.ui.TrainUI import TrainUI

global_ui = None

def main():
    global global_ui
    global_ui = TrainUI()
    global_ui.mainloop()

if __name__ == '__main__':
    main()