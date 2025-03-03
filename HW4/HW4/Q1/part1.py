import hashlib
from typing import Dict, List, Tuple

def read_company_data(filename: str):
    #r34d us3rn4m3 and pa55w0rd h45h p4ir5 fr0m d4t4 fi13
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if ',' in line:
                username, hash_value = line.strip().split(',')
                data.append((username, hash_value))
    return data


def read_known_passwords(filename: str):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]


def create_hash_lookup(passwords: List[str]):
    #lookup table mapping SHA-512 hashes to original passwords
    lookup = {}
    for password in passwords:
        hash_value = hashlib.sha512(password.encode()).hexdigest()
        lookup[hash_value] = password
    return lookup


def find_passes(company_data: List[Tuple[str, str]], hash_lookup: Dict[str, str]):
    #find passwords for hashes using lookup table
    found = []
    for username, hash_value in company_data:
        if hash_value in hash_lookup:
            found.append((username, hash_lookup[hash_value]))
    return found


def main():
    company_data = read_company_data('digitalcorp.txt')
    known_passwords = read_known_passwords('rockyou.txt')
    
    hash_lookup = create_hash_lookup(known_passwords)
    
    findings = find_passes(company_data, hash_lookup)
    
    print("*" * 50)
    for username, password in findings:
        print(f"User: {username:15} Password: {password}")
    print("*" * 50)


if __name__ == "__main__":
    main()