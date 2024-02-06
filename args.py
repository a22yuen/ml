import argparse

def main():
    parser = argparse.ArgumentParser(description='Example script with flags.')
    parser.add_argument('pos_arg', type=int, help='A positional argument')
    parser.add_argument('--opt_arg', type=int, help='An optional argument')
    parser.add_argument('--flag', action='store_true', help='A boolean flag')
    
    args = parser.parse_args()

    print(args.pos_arg)
    if args.opt_arg:
        print(f"Optional arg value: {args.opt_arg}")
    if args.flag:
        print("Flag was passed")

if __name__ == "__main__":
    main()
