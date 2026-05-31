from mjlab.scripts.play import main as mjlab_main
from .startup import startup

def main():
    startup(mjlab_main)

if __name__ == '__main__':
    main()
