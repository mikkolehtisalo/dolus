import secrets
import random
import tools
import argparse

wordlists = tools.read_files_to_byte_strings("../wordlists/")
special_chars = rb"!@#$%^&*()_-+={}[]|\:;\"'<>,.?/`"

def gen_random_bytes()-> bytes:
    """
    Generates strong & secure passwords via the secrets module.

    :return: Byte string containing the password
    """
    return secrets.token_bytes(random.randint(8,24))

def gen_random_urlsafe()-> bytes:
    """
    Generates strong & secure passwords via the secrets module.

    :return: Byte string containing the password
    """
    return bytes(secrets.token_urlsafe(random.randint(8,24)), "utf-8")

def gen_random_mnemonic() -> bytes:
    """
    Generates strong & secure passwords via mnenomics and some randomization.

    :return: Byte string containing the password
    """
    mnemonic = b""
    fnumber = random.randint(0,len(wordlists)-1)
    mlen = random.randint(5,10)
    for _ in range(mlen):
        mnumber = random.randint(0,2048-1)
        word = wordlists[fnumber][mnumber]
        # 25 % chance to capitalize the word
        if random.randint(0,100) > 75:
            word = tools.capitalize_byte_string(word)
        mnemonic += word
        # 25 % change to add a random number
        if random.randint(0,100) > 75:
            numb = str(random.randint(0,9))
            mnemonic += bytes(numb, "utf-8")
        # 10 % change to add a random special character
        if random.randint(0,100) > 90:
            schr = special_chars[random.randint(0,len(special_chars)-1)]
            mnemonic += schr.to_bytes()
    # 25 % change to finish the password with couple more random numbers
    if random.randint(0,100) > 75:
        for _ in range(random.randint(1,3)):
            numb = str(random.randint(0,9))
            mnemonic += bytes(numb, "utf-8")
     # 10 % change to add a random special character
    if random.randint(0,100) > 90:
        schr = special_chars[random.randint(0,len(special_chars)-1)]
        mnemonic += schr.to_bytes()
    return mnemonic

def gen_random_xkcd() -> bytes:
    """ 
    Generates strong & secure passwords pretty much like xkcd described.

    :return: Byte string containing the password
    """
    password = b""
    # Since it's not clear whether the words should have a space between them or not, 50 % chance for spaces
    space = random.randint(0,1) == 1
    fnumber = random.randint(0,len(wordlists)-1)
    for _ in range(random.randint(5,8)):
        mnumber = random.randint(0,2048-1)
        word = wordlists[fnumber][mnumber]
        password += word
        if space:
            password += b" "
    return password

def gen_file(fname: str, number: int):
    """
    Generates a file of strong passwords

    :fname: Name of the file to be generated
    :number: How many passwords should the file contain
    """
    with open(fname, "wb") as binary_file:
        for _ in range(number):
            x = random.randint(0,2)
            match x:
                case 0:
                    binary_file.write(gen_random_mnemonic())
                case 1:
                    binary_file.write(gen_random_urlsafe())
                case 2:
                    binary_file.write(gen_random_xkcd())    
                    
            binary_file.write(bytes("\n", "utf-8"))

def main():
    parser = argparse.ArgumentParser(prog='generate.py', description='Generates synthetic data for "good" passwords')
    parser.add_argument('output', help="File that gets written")
    parser.add_argument('lines', help="How many passwords to generate") 
    parser.parse_args()
    args = parser.parse_args()

    gen_file(args.output, int(args.lines))

if __name__ == "__main__":
    main()

