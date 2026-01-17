import pandas as pd
import numpy as np
import re
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.panel import Panel

console = Console()

class EmailDataProcessor:
    """Handles loading and processing of email dataset"""
    
    def __init__(self):
        self.console = Console()
    
    def parse_email_csv_state_machine(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Parse using state machine approach for complex CSV format with quotes and multiline fields.
        This handles the specific format of the provided email dataset.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split into records using email_id pattern as delimiter
            records = []
            current_record = []

            for line in content.splitlines():
                line = line.rstrip('\n')
                # New record starts with pattern: "<number>,
                if re.match(r'^"\d+,', line):
                    if current_record:
                        records.append('\n'.join(current_record))
                    current_record = [line]
                else:
                    if current_record:
                        current_record.append(line)

            if current_record:
                records.append('\n'.join(current_record))

            self.console.print(f"[bold green]✅ Found {len(records)} email records in the dataset[/bold green]")

            parsed_data = []

            # Patterns for identifying end fields
            thread_pattern = r'thread_\d{3}$'
            attachment_pattern = r'(TRUE|FALSE)$'
            timestamp_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$'

            for idx, record in enumerate(records):
                try:
                    # Remove outer quotes if present
                    if record.startswith('"') and record.endswith('"'):
                        record = record[1:-1]

                    # Split by commas, but we'll reconstruct intelligently
                    parts = record.split(',')

                    if len(parts) < 8:
                        self.console.print(f"[yellow]⚠️ Skipping record {idx+1}: Insufficient parts ({len(parts)})[/yellow]")
                        continue

                    # Find thread_id from the end
                    thread_id = None
                    has_attachment = None
                    timestamp = None

                    # Look for thread_id (last field should match pattern)
                    if re.search(thread_pattern, parts[-1].strip()):
                        thread_id = parts[-1]
                        remaining = parts[:-1]
                    else:
                        self.console.print(f"[yellow]⚠️ Skipping record {idx+1}: Invalid thread_id format[/yellow]")
                        continue

                    # Look for has_attachment (should be TRUE/FALSE)
                    if remaining and re.search(attachment_pattern, remaining[-1].strip()):
                        has_attachment = remaining[-1]
                        remaining = remaining[:-1]
                    else:
                        self.console.print(f"[yellow]⚠️ Skipping record {idx+1}: Invalid attachment format[/yellow]")
                        continue

                    # Look for timestamp (ISO format)
                    if remaining and re.search(timestamp_pattern, remaining[-1].strip()):
                        timestamp = remaining[-1]
                        remaining = remaining[:-1]
                    else:
                        self.console.print(f"[yellow]⚠️ Skipping record {idx+1}: Invalid timestamp format[/yellow]")
                        continue

                    # First 4 fields should be email_id, sender_email, sender_name, subject
                    if len(remaining) < 4:
                        self.console.print(f"[yellow]⚠️ Skipping record {idx+1}: Insufficient remaining parts ({len(remaining)})[/yellow]")
                        continue

                    email_id = remaining[0].strip('"')
                    sender_email = remaining[1].strip('"')
                    sender_name = remaining[2].strip('"')
                    subject = remaining[3].strip('"')

                    # Everything else is body
                    body_parts = remaining[4:]
                    body = ','.join(body_parts)

                    # Clean up escaped quotes in body (double quotes are escaped as double double quotes)
                    body = body.replace('""', '"')
                    # Remove leading/trailing quotes if present
                    if body.startswith('"') and body.endswith('"'):
                        body = body[1:-1]
                    # Handle newlines that were split by the CSV format
                    body = body.replace('"\\n"', '\n').replace('\\"', '"')

                    parsed_data.append([
                        email_id, sender_email, sender_name, subject,
                        body, timestamp, has_attachment, thread_id
                    ])

                except Exception as e:
                    self.console.print(f"[red]❌ Error parsing record {idx+1}: {str(e)}[/red]")
                    continue

            if not parsed_data:
                raise ValueError("No valid records were parsed from the dataset")

            columns = ['email_id', 'sender_email', 'sender_name', 'subject', 'body', 'timestamp', 'has_attachment', 'thread_id']
            df = pd.DataFrame(parsed_data, columns=columns)

            # Clean data types
            df['email_id'] = df['email_id'].astype(int)
            df['has_attachment'] = df['has_attachment'].map({'TRUE': True, 'FALSE': False})
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            self.console.print(f"[bold green]✅ Successfully parsed {len(df)} email records[/bold green]")
            self.console.print(f"[bold blue]📊 Unique threads: {df['thread_id'].nunique()}[/bold blue]")
            self.console.print(f"[bold blue]📊 Email ID range: {df['email_id'].min()} to {df['email_id'].max()}[/bold blue]")

            return df

        except Exception as e:
            self.console.print(f"[bold red]❌ Error in parsing function: {str(e)}[/bold red]")
            self.console.print("[yellow]Falling back to sample dataset...[/yellow]")
            return None
    
    # def create_sample_dataset(self, num_emails: int = 100) -> pd.DataFrame:
    #     """Create a comprehensive sample dataset matching the expected format"""
    #     np.random.seed(42)
        
    #     data = {
    #         'email_id': list(range(1, num_emails + 1)),
    #         'sender_name': np.random.choice([
    #             'John Smith', 'GitHub Notifications', 'Billing Department', 'Sarah Johnson', 'System Alert',
    #             'Marketing Team', 'HR Department', 'Customer Support', 'Project Manager', 'Team Lead',
    #             'Tech Newsletter', 'IT Security', 'Finance Team', 'Sales Representative', 'Recruiter'
    #         ], num_emails),
    #         'sender_email': np.random.choice([
    #             'john.smith@company.com', 'noreply@github.com', 'billing@service.com', 'sarah.johnson@company.com', 'noreply@system.com',
    #             'marketing@company.com', 'hr@company.com', 'support@company.com', 'pm@company.com', 'teamlead@company.com',
    #             'newsletter@tech.com', 'security@company.com', 'finance@company.com', 'sales@company.com', 'careers@company.com'
    #         ], num_emails),
    #         'subject': np.random.choice([
    #             'Project Alpha - Budget Approval Needed', 'Your Weekly Tech Digest', 'Invoice #INV-2024-0156', 
    #             'Project Status Update - Phoenix', 'Security Alert: Failed Login Attempts',
    #             'Q1 Marketing Campaign Results', 'Updated PTO Policy Effective Next Month', 
    #             'Re: Support Ticket #4532 - Resolved', 'Sprint Planning Meeting - Monday 10 AM'
    #         ], num_emails),
    #         'body': np.random.choice([
    #             'Hi Team,\n\nI need urgent approval for the Project Alpha budget increase. We\'re looking at an additional $50K for Q1 due to unexpected infrastructure costs.\n\nCan we schedule a meeting this week to discuss? The deadline for approval is Friday.\n\nBest regards,\nJohn',
    #             'Hello Subscriber,\n\nThis week\'s highlights:\n- New feature launch: Dark mode\n- Community spotlight: User success story\n- Upcoming webinar: Advanced analytics techniques\n\nRead more on our website.\n\nTech Weekly Team',
    #             'Hello,\n\nYour monthly subscription invoice is attached. Payment is due by the end of the month. Total amount: $49.99\n\nThank you for your business!\n\nSupport Team',
    #             'Here is the weekly update on Project Phoenix. We have completed phase 1 and moving to phase 2. Key accomplishments:\n- User authentication integrated\n- Database schema finalized\n- Initial UI mockups approved\n\nNext steps include API development and testing.\n\nSarah Johnson',
    #             'ALERT: Server experiencing high CPU usage (95%). Please investigate immediately. This is affecting production services. Contact on-call engineer if unresolved within 30 minutes.'
    #         ], num_emails),
    #         'timestamp': pd.date_range('2024-01-01', periods=num_emails, freq='H'),
    #         'has_attachment': np.random.choice([True, False], num_emails, p=[0.3, 0.7]),
    #         'thread_id': [f'thread_{i:03d}' for i in np.random.randint(1, 50, num_emails)]
    #     }
        
        df = pd.DataFrame(data)
        self.console.print(f"[bold green]✅ Created sample dataset with {len(df)} emails[/bold green]")
        return df
    
    def load_dataset(self, file_path: str = "dataset_emails - Sheet1.csv") -> pd.DataFrame:
        """Load and process the email dataset"""
        try:
            df = self.parse_email_csv_state_machine(file_path)
            if df is None or df.empty:
                self.console.print("[yellow]⚠️ Using sample dataset instead[/yellow]")
                df = self.create_sample_dataset()
            return df
        except Exception as e:
            self.console.print(f"[red]❌ Error loading dataset: {str(e)}[/red]")
            self.console.print("[yellow]⚠️ Using sample dataset instead[/yellow]")
            return self.create_sample_dataset()
    
    def preprocess_email(self, email_row: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single email for analysis"""
        return {
            'email_id': email_row['email_id'],
            'sender_email': email_row['sender_email'],
            'sender_name': email_row['sender_name'],
            'subject': str(email_row['subject']).strip(),
            'body': str(email_row['body']).strip(),
            'timestamp': email_row['timestamp'],
            'has_attachment': email_row['has_attachment'],
            'thread_id': email_row['thread_id'],
            'full_text': f"{email_row['subject']} {email_row['body']}".strip()
        }
