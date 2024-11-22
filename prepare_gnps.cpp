#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <time.h>
#include <math.h>
#include <chrono>

using namespace std;

int main() {
    ifstream ifs("./data/gnps_data/cleaned_spectra.mgf");
    ofstream ofs("./data/processed_gnps_data/cleaned_spectra_processed.csv");
    string line;

    //time_t t = time(0);
    auto start_time = chrono::system_clock::now();

    int count = 0;
    while (getline(ifs, line)) {
        if (line.find("BEGIN IONS") != string::npos) {
            string ion_mode, adduct, smiles, spectrum_id, ionization_type;
            string ms_level, precursor_mz, parent_mass;
            ion_mode = "None";
            adduct = "None";
            smiles = "None";
            spectrum_id = "None";
            ionization_type = "None";
            ms_level = "None";
            precursor_mz = "None";
            parent_mass = "None";

            while (getline(ifs, line) && line.find("=") != string::npos) {
                if (line.find("ADDUCT") != string::npos) {
                    adduct = line.substr(line.find("=") + 1);
                }
                else if (line.find("SMILES") != string::npos) {
                    smiles = line.substr(line.find("=") + 1);
                } 
                else if (line.find("IONMODE") != string::npos) {
                    ion_mode = line.substr(line.find("=") + 1);
                }
                else if (line.find("SPECTRUM_ID") != string::npos) {
                    spectrum_id = line.substr(line.find("=") + 1);
                }
                else if (line.find("MS_IONISATION") != string::npos) {
                    ionization_type = line.substr(line.find("=") + 1);
                }
                else if (line.find("MS_LEVEL") != string::npos) {
                    ms_level = line.substr(line.find("=") + 1);
                }
                else if (line.find("PRECURSOR_MZ") != string::npos) {
                    precursor_mz = line.substr(line.find("=") + 1);
                }
                else if (line.find("PARENT_MASS") != string::npos) {
                    parent_mass = line.substr(line.find("=") + 1);
                }

            }
            ofs << spectrum_id << "," << smiles << "," << adduct;
            ofs << "," << ion_mode;
            ofs << "," << ionization_type << "," << ms_level << ",";
            ofs << precursor_mz << "," << parent_mass << ",";
            ofs << "\"[";
            while (getline(ifs, line) 
                    && line.find("END IONS") == string::npos ) {
                    vector<pair<double, double>> mass_charge_pairs;
                    istringstream iss(line);
                    double mass, charge;
                    iss >> mass >> charge;
                    
                    mass_charge_pairs.push_back(make_pair(mass, charge));
                    for (const auto& pair : mass_charge_pairs) {
                        ofs << "[" << pair.first << "," << pair.second << "],";
                    }
            }
            ofs << "]\"" << endl;
            count++;
            if (count % 1000 == 0) {
                auto now = chrono::system_clock::now();
                auto duration = now.time_since_epoch() - start_time.time_since_epoch();
                auto milliseconds = chrono::duration_cast<chrono::milliseconds>(duration).count();
                cout << "Processed " << count << " spectra" << endl;
                cout << "Time: " << milliseconds << " ms" << endl;
            }
        }
    }
    cout << "Finished processing " << count << " spectra" << endl;
    auto now = chrono::system_clock::now();
    auto duration = now.time_since_epoch() - start_time.time_since_epoch();
    auto milliseconds = chrono::duration_cast<chrono::milliseconds>(duration).count();
    cout << "Time: " << milliseconds << " ms" << endl;
    return 0;
}