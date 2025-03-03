import hashlib
import itertools
from typing import List, Tuple

def read_company_data(filename: str):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if ',' in line:
                username, salt, hash_value = line.strip().split(',')
                data.append((username, salt, hash_value))
    return data


def read_password_list(filename: str):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]


def hash_permutation(x_prev: str, plain_password: str, salt: str, iteration: int, order: Tuple[int, int, int]):
    elements = [x_prev, plain_password, salt]
    ordered_elements = ''.join(elements[i] for i in order)
    return hashlib.sha512(ordered_elements.encode()).hexdigest()


def try_key_stretching(password: str, salt: str, target_hash: str, max_iterations: int = 2000):
    for order in itertools.permutations(range(3)):
        x_prev = ""
        for iteration in range(1, max_iterations + 1):
            x_prev = hash_permutation(x_prev, password, salt, iteration, order)
            if x_prev == target_hash:
                return True, password
    return False, ""


def find_passes(company_data: List[Tuple[str, str, str]], known_passwords: List[str]):
    findings = []

    for username, salt, target_hash in company_data:
        for password in known_passwords:
            is_match, cracked_password = try_key_stretching(password, salt, target_hash)
            if is_match:
                findings.append((username, cracked_password))
                break
    
    return findings


def main():
    company_data = read_company_data('stretched-digitalcorp.txt')
    known_passwords = read_password_list('rockyou.txt')

    findings = find_passes(company_data, known_passwords)

    print("*" * 50)
    for username, password in findings:
        print(f"User: {username:15} password: {password}")
    print("*" * 50)

if __name__ == "__main__":
    main()

