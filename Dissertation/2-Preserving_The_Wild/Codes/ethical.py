# Example: Data anonymization function
def anonymize_data(data):
    # Implement anonymization techniques here (e.g., remove personal identifiers)
    anonymized_data = data.copy()
    anonymized_data['user_id'] = None  # Anonymize user ID
    return anonymized_data

# Example: Access control function
def check_access(user_role):
    allowed_roles = ['researcher', 'admin']
    if user_role in allowed_roles:
        return True
    else:
        return False

# Example usage
data = {'user_id': '12345', 'sensor_data': [0.1, 0.2, 0.3]}
anonymized_data = anonymize_data(data)
user_role = 'researcher'
if check_access(user_role):
    print("Access granted.")
else:
    print("Access denied.")
