# Virtual clients για το Fall Guardian

Simulator σε Python που τρέχει πολλά εικονικά ρολόγια. Κάθε ρολόι «παίζει»
εγγραφές IMU (SisFall, MobiFall ή συνθετικές), τρέχει πάνω τους έναν ανιχνευτή
πτώσης όπως θα έκανε το ESP32-C6, και μιλάει MQTT με το
[πρωτόκολλο v1](PROTOCOL.md). Ένας monitor στην πλευρά του server ακούει όλα
τα ρολόγια, επιβεβαιώνει συναγερμούς και μετράει πόσο σωστά ανιχνεύτηκαν.

```
fallsim/
  protocol.py   topics, QoS, κατασκευή μηνυμάτων
  datasets.py   φόρτωση SisFall / MobiFall v2, συνθετικές εγγραφές, resampling
  detector.py   ανιχνευτής κατωφλίου (baseline), interface για μοντέλο TinyML
  watch.py      ένα virtual ρολόι
  simulate.py   CLI: πολλά ρολόγια
  monitor.py    CLI: server-side παρακολούθηση και στατιστικά
tests/          pytest (offline + end-to-end με ενσωματωμένο broker)
```

## Γρήγορη εκκίνηση

```bash
cd virtual-clients
python3 -m pip install -r requirements.txt

# broker: mosquitto από docker, ή όποιον έχεις ήδη
docker compose up -d

# τερματικό 1: server-side monitor
python3 -m fallsim.monitor --auto-ack --summary-every 10

# τερματικό 2: 20 ρολόγια με συνθετικά δεδομένα, 10x πιο γρήγορα από πραγματικό χρόνο
python3 -m fallsim.simulate --devices 20 --speed 10 --duration 600
```

Με Ctrl-C ο monitor τυπώνει σύνοψη, π.χ. από ένα τρέξιμο 20 ρολογιών:

```
"truth": {"tp": 95, "fn": 0, "fp": 54, "tn": 228, "sensitivity": 1.0, "specificity": 0.8085,
          "per_activity": {"FALL": {"n": 95, "detected": 95}, "JMP": {"n": 65, "detected": 54}, ...}}
```

Τα ψευδώς θετικά είναι σχεδόν όλα άλματα (`JMP`): κρούση και μετά ακινησία
μοιάζει με πτώση για έναν ανιχνευτή κατωφλίου. Αυτό είναι ακριβώς το κενό που
πρέπει να κλείσει το μοντέλο TinyML.

## Datasets

Τα datasets δεν μπαίνουν στο repo. Κατέβασέ τα και δώσε τον φάκελο με `--data-root`.

**SisFall** (Sucerquia et al., 2017): 38 άτομα (23 νέοι `SA`, 15 ηλικιωμένοι
`SE`), 15 τύποι πτώσης `F01–F15`, 19 δραστηριότητες `D01–D19`, 200 Hz, στη μέση.
Ο parser μετατρέπει τα raw counts σε g (ADXL345 ±16 g/13 bit, ή MMA8451Q με
`--sisfall-accel mma8451q`) και °/s (ITG3200).

```bash
python3 -m fallsim.simulate --dataset sisfall --data-root ~/data/SisFall_dataset \
    --devices 38 --speed 5 --subjects SE01,SE02,SE06
```

**MobiFall v2.0** (Vavoulas et al., 2014): smartphone στην τσέπη, ~87 Hz με
ακανόνιστα timestamps. Πτώσεις `FOL, FKL, BSC, SDL`. Ο parser διαβάζει τα
`*_acc_*.txt` (m/s²) και τα αντίστοιχα `*_gyro_*.txt` (rad/s) και τα φέρνει σε
ομοιόμορφο πλέγμα.

```bash
python3 -m fallsim.simulate --dataset mobifall --data-root ~/data/MobiFall_Dataset_v2.0 --devices 24
```

**Συνθετικά** (προεπιλογή): ακινησία, περπάτημα, τρέξιμο, κάθισμα, άλμα και
πτώσεις (ελεύθερη πτώση, κρούση 3–6 g, ξάπλωμα). Είναι χοντροκομμένα επίτηδες:
κάνουν για να δοκιμαστεί η αλυσίδα MQTT χωρίς downloads, όχι για να αξιολογηθεί
μοντέλο.

Δύο επιφυλάξεις για τη διπλωματική: και τα δύο datasets είναι από αισθητήρα στη
μέση ή στην τσέπη, όχι στον καρπό, άρα η κατανομή διαφέρει από αυτό που θα δει
το ρολόι. Και οι parsers έχουν γραφτεί πάνω στην τεκμηριωμένη μορφή των αρχείων
και δοκιμαστεί με αρχεία της ίδιας μορφής, όχι ακόμη με τα πραγματικά
downloads.

## Χρήσιμες επιλογές

| Επιλογή | Τι κάνει |
|---|---|
| `--devices N` | πόσα ρολόγια (`vw-0000` …) |
| `--speed X` | ταχύτητα αναπαραγωγής, οι χρονοσφραγίδες ακολουθούν τον χρόνο της προσομοίωσης |
| `--duration S` | δευτερόλεπτα προσομοίωσης, αλλιώς μέχρι Ctrl-C |
| `--fs 50` | ρυθμός IMU του ρολογιού, όλα τα δεδομένα γίνονται resample σε αυτόν |
| `--imu-stream` | στέλνει και τα ακατέργαστα δείγματα στο `.../imu` |
| `--only falls\|adl` | μόνο πτώσεις ή μόνο καθημερινές δραστηριότητες |
| `--subjects SA01,SE06` | μόνο συγκεκριμένα άτομα |
| `--host --port --username --password --tls` | broker |
| `--summary-json out.json` | αποθήκευση σύνοψης |

Ο monitor δέχεται τα ίδια `--host/--port/...`, συν `--auto-ack` (στέλνει
`cmd ack` σε κάθε πτώση) και `--quiet`.

Σε δοκιμή στο cloud container, 200 ρολόγια σε πραγματικό χρόνο με
`--imu-stream` (50 Hz, ~400 μηνύματα/s) κράτησαν τον ρυθμό με έναν broker amqtt.

## Σύνδεση με το μοντέλο TinyML

Ένας ανιχνευτής είναι οτιδήποτε έχει `name` και `update(acc, gyr) -> Detection | None`
(βλ. `detector.py`). Όταν το μοντέλο από το pipeline εκπαίδευσης βγει σε
`.tflite`, μια κλάση `TFLiteDetector` που κρατά παράθυρο δειγμάτων και καλεί τον
interpreter μπαίνει στο `DETECTORS` και επιλέγεται με `--detector`. Έτσι το ίδιο
μοντέλο τρέχει πρώτα σε εκατοντάδες virtual ρολόγια και μετά στο πραγματικό.

## Tests

```bash
python3 -m pip install pytest amqtt
python3 -m pytest
```

Το end-to-end test σηκώνει broker amqtt μέσα στη διεργασία, τρέχει 4 ρολόγια
και ελέγχει ότι ο monitor βλέπει κάθε μήνυμα και ότι τα `ack` φτάνουν πίσω.
