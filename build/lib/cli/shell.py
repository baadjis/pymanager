import cmd
import argparse
import sys
import os

def help_builder(text):
    def wrapper(args):
        text()
        return False

    return wrapper


class CostumedCommand(cmd.Cmd, object):
    prompt = "$"

    def cmdloop(self, intro=None):
        while True:
            try:
                super(CostumedCommand, self).cmdloop(intro)
                break
            except KeyboardInterrupt:
                print("^C\n   bye...")
                break

    def preloop(self):
        """Initialization before prompting user for commands.
           Despite the claims in the Cmd documentaion, Cmd.preloop() is not a stub.
        """
        cmd.Cmd.preloop(self)  ## sets up command completion
        self._hist = []  ## No history yet

    def precmd(self, line):
        """ This method is called after the line has been input but before
            it has been interpreted. If you want to modifdy the input line
            before execution (for example, variable substitution) do it here.
        """
        self._hist += [line.strip()]
        return line

    def postloop(self):
        """Take care of any unfinished business.
           Despite the claims in the Cmd documentaion, Cmd.postloop() is not a stub.
        """
        cmd.Cmd.postloop(self)  ## Clean up command completion
        print("   Bye...")

    ## Command definitions ##
    def do_hist(self, args):
        """Print a list of commands that have been entered"""
        print("\n".join(self._hist))

    def do_exit(self, args):
        """Exits from the console"""
    
        return True


class Cli():
    def __init__(self, name):
        self.name = name
        self.commands = []

    def command(self, cmd_args=None):
        c_args = {} if cmd_args is None else cmd_args

        def wrapper(func):

            parser = argparse.ArgumentParser(prog=func.__name__, description=func.__doc__, add_help=False)
            for xk, xv in c_args.items():
                parser.add_argument(xk, **xv)

            def anon(self, args):
                try:
                    out = parser.parse_args(args.split())
                    func(**vars(out))
                except SystemExit as e:
                    pass

            self.commands.append({
                "name": func.__name__,
                "method": anon,
                "doc": parser.print_help,
                "parser": parser
            })
            return func

        return wrapper

    def get_shell(self):

        methods = self.commands

        d = {}
        for func_desc in methods:
            d['do_' + func_desc['name']] = func_desc['method']
            d['help_' + func_desc['name']] = help_builder(func_desc['doc'])

        return type(self.__class__.__name__ + '_shell', (CostumedCommand,), d)()

    def run(self):
        the_args = sys.argv  # first arg is file
        
        # Run shell if no second arguments
        if len(the_args) == 1:
            shell = self.get_shell()
            shell.prompt = self.name + "$"
            shell.cmdloop()
            return 0

        # Get function name
        func_name = the_args[1]

        # First argument is function name
        for func_desc in self.commands:
            if func_name == func_desc['name']:
                func_desc['method'](None, " ".join(the_args[2:]))
                return 0

        print('No such command')


CLI = Cli("my")


@CLI.command(cmd_args={'price': {'type': float, 'default': 0, 'nargs': '?'}})
def pricer(price: float):
    
    print("price is $" + '{0:.2f}'.format(price))


if __name__ == '__main__':
    CLI.run()
