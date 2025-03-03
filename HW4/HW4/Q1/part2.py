import hashlib
from typing import List, Tuple

def read_company_data(filename: str):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if ',' in line:
                username, salt, hash_value = line.strip().split(',')
                data.append((username, salt, hash_value))
    return data


def read_known_passwords(filename: str):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]


def try_hack_hash(password: str, salt: str, target_hash: str):
    #try appnd and prepen.
    prepended = hashlib.sha512((salt + password).encode()).hexdigest()
    if prepended == target_hash:
        return True, True
        
    appended = hashlib.sha512((password + salt).encode()).hexdigest()
    if appended == target_hash:
        return True, False
    
    return False, False


def find_passes(company_data: List[Tuple[str, str, str]], known_passwords: List[str]):
    findings = []
    
    for username, salt, hash_value in company_data:
        for password in known_passwords:
            is_match, is_prepended = try_hack_hash(password, salt, hash_value)
            if is_match:
                findings.append((username, password, is_prepended))
                break
    
    return findings


def main():
    company_data = read_company_data('salty-digitalcorp.txt')
    known_passwords = read_known_passwords('rockyou.txt')
    
    findings = find_passes(company_data, known_passwords)
    
    print("*" * 50)
    for username, password, is_prepended in findings:
        salt_position = "prepended" if is_prepended else "appended"
        print(f"User: {username:15} password: {password:20} (Salt {salt_position})")
    print("*" * 50)
    

if __name__ == "__main__":
    main()