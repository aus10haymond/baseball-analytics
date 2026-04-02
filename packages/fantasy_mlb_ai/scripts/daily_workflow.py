"""
Automated Daily Workflow

Orchestrates the complete daily fantasy baseball workflow:
1. Fetch roster and game data
2. Generate pitcher-aware projections
3. Save recommendations
4. Track predictions
5. Send notifications (optional)

Run this script daily at 9 AM during baseball season.
"""

import sys
import logging
from pathlib import Path
from datetime import date, datetime, timedelta

# Add parent to path for imports
parent_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(parent_dir))

from fantasy_mlb_ai.pitcher_aware_projections import PitcherAwareEngine
from fantasy_mlb_ai.prediction_tracker import PredictionTracker
from fantasy_mlb_ai.accuracy_reporter import AccuracyReporter
from fantasy_mlb_ai.fetch_daily_data import main as fetch_data
import pandas as pd

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"daily_workflow_{date.today()}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def send_email_notification(recommendations_df: pd.DataFrame, subject: str):
    """
    Send email notification with recommendations (optional).

    Args:
        recommendations_df: DataFrame with recommendations
        subject: Email subject line

    Note: Requires SMTP configuration in environment variables
    """
    import os
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    # Check if email is configured
    smtp_host = os.getenv('SMTP_HOST')
    smtp_port = os.getenv('SMTP_PORT', '587')
    smtp_user = os.getenv('SMTP_USER')
    smtp_pass = os.getenv('SMTP_PASSWORD')
    email_to = os.getenv('EMAIL_TO')

    if not all([smtp_host, smtp_user, smtp_pass, email_to]):
        logger.info("Email not configured - skipping notification")
        return

    try:
        # Create email
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = email_to
        msg['Subject'] = subject

        # Build HTML body
        html = f"""
        <html>
        <head>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h2>Fantasy Baseball Daily Recommendations</h2>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {recommendations_df.to_html(index=False)}
        </body>
        </html>
        """

        msg.attach(MIMEText(html, 'html'))

        # Send email
        with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        logger.info(f"Email sent to {email_to}")

    except Exception as e:
        logger.error(f"Failed to send email: {e}")


def send_error_alert(error_msg: str):
    """
    Send error alert notification.

    Args:
        error_msg: Error message to send
    """
    logger.error(f"WORKFLOW ERROR: {error_msg}")
    # Could integrate with Slack, Discord, SMS, etc. here
    # For now, just log it


def main():
    """
    Main workflow execution.
    """
    logger.info("="*70)
    logger.info(f"Starting Daily Fantasy Workflow - {date.today()}")
    logger.info("="*70)

    today_str = date.today().strftime('%Y-%m-%d')
    yesterday_str = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        # Step 1: Fetch daily data
        logger.info("\n[1/6] Fetching ESPN roster and MLB games...")
        try:
            fetch_data()  # Runs fetch_daily_data.py
            logger.info("✓ Data fetch complete")
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            # Continue anyway - might have cached data

        # Load data
        data_dir = Path(__file__).parent.parent / "data"
        roster_df = pd.read_csv(data_dir / "my_roster.csv")
        games_df = pd.read_csv(data_dir / "mlb_games_today.csv")

        logger.info(f"  Loaded {len(roster_df)} players from roster")
        logger.info(f"  Found {len(games_df)} games today")

        # Step 2: Initialize projection engine
        logger.info("\n[2/6] Initializing pitcher-aware projection engine...")
        engine = PitcherAwareEngine()

        if not engine.ml_available:
            raise Exception("ML models not available")

        logger.info("✓ Projection engine ready")

        # Step 3: Generate projections
        logger.info("\n[3/6] Generating pitcher-aware projections...")

        team_name_map = {
            'Tor': 'Toronto Blue Jays', 'Phi': 'Philadelphia Phillies',
            'Ari': 'Arizona Diamondbacks', 'NYY': 'New York Yankees',
            'SD': 'San Diego Padres', 'Mil': 'Milwaukee Brewers',
            'LAA': 'Los Angeles Angels', 'NYM': 'New York Mets',
            'Bos': 'Boston Red Sox', 'Tex': 'Texas Rangers',
            'Atl': 'Atlanta Braves', 'Bal': 'Baltimore Orioles',
            'ChC': 'Chicago Cubs', 'ChW': 'Chicago White Sox',
            'Cin': 'Cincinnati Reds', 'Cle': 'Cleveland Guardians',
            'Col': 'Colorado Rockies', 'Det': 'Detroit Tigers',
            'Hou': 'Houston Astros', 'KC': 'Kansas City Royals',
            'LAD': 'Los Angeles Dodgers', 'Mia': 'Miami Marlins',
            'Min': 'Minnesota Twins', 'Oak': 'Oakland Athletics',
            'Pit': 'Pittsburgh Pirates', 'Sea': 'Seattle Mariners',
            'SF': 'San Francisco Giants', 'StL': 'St. Louis Cardinals',
            'TB': 'Tampa Bay Rays', 'Was': 'Washington Nationals'
        }

        projections = engine.get_roster_matchup_projections(
            roster_df,
            team_name_map,
            default_pa=4
        )

        # Count matchup types
        matchup_counts = projections['pa_matchup_type'].value_counts()
        logger.info(f"✓ Generated {len(projections)} projections")
        for matchup_type, count in matchup_counts.items():
            logger.info(f"    {matchup_type}: {count}")

        # Step 4: Save recommendations
        logger.info("\n[4/6] Saving recommendations...")
        output_dir = data_dir / "recs"
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / f"recommendations-{today_str}.csv"
        projections.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to {output_path}")

        # Step 5: Track predictions
        logger.info("\n[5/6] Logging predictions for accuracy tracking...")
        tracker = PredictionTracker()

        predictions_logged = 0
        for _, row in projections.iterrows():
            if row['pa_projection'] is not None and row['pa_projection'] > 0:
                tracker.log_prediction(
                    date_str=today_str,
                    player_name=row['name'],
                    projected=row['pa_projection'],
                    confidence=row['pa_confidence'],
                    matchup_type=row['pa_matchup_type'],
                    pitcher=row.get('opponent_pitcher'),
                    team=row.get('proTeam'),
                    position=row.get('position'),
                    was_started=(row.get('lineupSlot') not in ['BE', 'IL'])
                )
                predictions_logged += 1

        tracker.close()
        logger.info(f"✓ Logged {predictions_logged} predictions")

        # Step 6: Generate accuracy report for yesterday
        logger.info(f"\n[6/6] Generating accuracy report for {yesterday_str}...")
        try:
            with AccuracyReporter() as reporter:
                daily_report = reporter.generate_daily_report(yesterday_str)
                print("\n" + daily_report)

                # Save report to file
                report_dir = data_dir / "reports"
                report_dir.mkdir(exist_ok=True)
                report_path = report_dir / f"accuracy_report_{yesterday_str}.txt"

                with open(report_path, 'w') as f:
                    f.write(daily_report)

                logger.info(f"✓ Accuracy report saved to {report_path}")

        except Exception as e:
            logger.warning(f"Could not generate accuracy report: {e}")

        # Step 7: Send notification (optional)
        logger.info("\n[7/7] Sending notifications...")
        try:
            # Prepare summary for notification
            active_players = projections[
                (projections['pa_projection'].notna()) &
                (projections['pa_projection'] > 0)
            ].sort_values('pa_projection', ascending=False)

            summary = active_players[[
                'name', 'position', 'pa_projection',
                'pa_confidence', 'opponent_pitcher', 'pa_matchup_type'
            ]].head(20)

            send_email_notification(
                summary,
                f"Fantasy Baseball Recommendations - {today_str}"
            )

        except Exception as e:
            logger.warning(f"Notification failed: {e}")

        logger.info("\n" + "="*70)
        logger.info("✓ Daily workflow complete!")
        logger.info("="*70)

        # Print summary
        print("\n" + "="*70)
        print(f"DAILY RECOMMENDATIONS - {today_str}")
        print("="*70)
        print("\nTop 10 Projections:")
        top_10 = projections.nlargest(10, 'pa_projection')[[
            'name', 'position', 'pa_projection', 'pa_confidence', 'opponent_pitcher'
        ]]
        print(top_10.to_string(index=False))
        print("\n" + "="*70)

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        send_error_alert(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
