import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Database Connection
# ----------------------------------------------------------
def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="arrg_db",
        user="postgres",
        password="postgres",
        port="5432"
    )

# ----------------------------------------------------------
# Insert New Patient Record
# ----------------------------------------------------------
def save_user_details(user_data):
    """
    Inserts a new patient record.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        insert_query = """
            INSERT INTO patient_records 
            (patient_id, name, age, gender, date_of_scan, symptoms, 
             family_history, head_injury_notes, other_conditions, 
             pred_label, confidence, dicom_filename)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """

        cur.execute(insert_query, (
            user_data["patient_id"],
            user_data["name"],
            user_data["age"],
            user_data["gender"],
            user_data["date_of_scan"],
            user_data["symptoms"],
            user_data["family_history"],
            user_data["head_injury_notes"],
            user_data["other_conditions"],
            user_data.get("pred_label"),
            user_data.get("confidence"),
            user_data.get("dicom_filename")
        ))

        conn.commit()
        cur.close()
        logger.info("Inserted patient_id=%s", user_data["patient_id"])
        return user_data["patient_id"]

    except Exception as e:
        logger.exception("Error in save_user_details:")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

# ----------------------------------------------------------
# Update Prediction Details
# ----------------------------------------------------------
def update_prediction_details(patient_id, pred_label, confidence, dicom_filename):
    """
    Updates the prediction fields for a patient.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        update_query = """
            UPDATE patient_records
            SET pred_label = %s,
                confidence = %s,
                dicom_filename = %s
            WHERE patient_id = %s;
        """
        cur.execute(update_query, (pred_label, confidence, dicom_filename, patient_id))
        conn.commit()
        rowcount = cur.rowcount
        cur.close()
        logger.info("Updated prediction details for patient_id=%s (rows=%s)", patient_id, rowcount)
        return rowcount

    except Exception as e:
        logger.exception("Error in update_prediction_details:")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

# ----------------------------------------------------------
# Fetch All Patients
# ----------------------------------------------------------
def get_all_patients():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patient_records ORDER BY timestamp ASC;")  # ✅ ascending order
    records = cur.fetchall()
    cur.close()
    conn.close()
    return records

