import streamlit as st
import datetime
import json
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import base64
import csv

def create_download_link(file_buffer, filename, link_text):
    """Create a download link for a file buffer."""
    b64 = base64.b64encode(file_buffer.getvalue()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'

def main():
    st.set_page_config(page_title="MAASP Calculator - ISO 16530-2", layout="wide")
    
    # Initialize session state
    if 'calculation_history' not in st.session_state:
        st.session_state.calculation_history = []
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'detailed_results' not in st.session_state:
        st.session_state.detailed_results = {}
    # Initialize derating_entries and apply_derating
    if 'derating_entries' not in st.session_state:
        st.session_state.derating_entries = {
            'temp_factor': 0.95,
            'service_factor': 0.90,
            'mfg_factor': 0.85,
            'design_factor': 0.80,
            'install_factor': 0.95,
            'env_factor': 0.90
        }
    if 'apply_derating' not in st.session_state:
        st.session_state.apply_derating = True
    
    st.title("MAASP Calculator - ISO 16530-2")
    
    # Create tabs
    tabs = st.tabs(["Well Information", "Calculator", "Derating Factors", "Results", "Calculation History"])
    
    with tabs[0]:
        create_well_info_tab()
    
    with tabs[1]:
        create_calculator_tab()
    
    with tabs[2]:
        create_derating_tab()
    
    with tabs[3]:
        create_results_tab()
    
    with tabs[4]:
        create_history_tab()

def create_well_info_tab():
    st.header("Well Information")
    st.session_state.well_info = {}
    
    col1, col2, col3 = st.columns(3)
    well_fields = [
        ("Well Name", "well_name"),
        ("Field", "field"),
        ("Operator", "operator"),
        ("Country", "country"),
        ("Rig", "rig"),
        ("Date", "date")
    ]
    
    for i, (label, key) in enumerate(well_fields):
        with [col1, col2, col3][i % 3]:
            if key == "date":
                st.session_state.well_info[key] = st.text_input(label, value=datetime.datetime.now().strftime("%Y-%m-%d"), key=f"well_info_{key}")
            else:
                st.session_state.well_info[key] = st.text_input(label, value="", key=f"well_info_{key}")

def create_calculator_tab():
    st.header("Calculator")
    
    # Annulus tabs
    calc_tabs = st.tabs(["A Annulus", "B Annulus", "C Annulus", "D Annulus"])
    
    with calc_tabs[0]:
        create_a_annulus_tab()
    with calc_tabs[1]:
        create_b_annulus_tab()
    with calc_tabs[2]:
        create_c_annulus_tab()
    with calc_tabs[3]:
        create_d_annulus_tab()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Calculate MAASP", key="calc_maasp_button"):
            calculate_all()
        if st.button("Clear All", key="clear_all_button"):
            clear_all()
    with col2:
        save_config()
        load_config()

def create_a_annulus_tab():
    if 'a_config' not in st.session_state:
        st.session_state.a_config = "long_casing"
    if 'a_entries' not in st.session_state:
        st.session_state.a_entries = {}
    if 'a_checkboxes' not in st.session_state:
        st.session_state.a_checkboxes = {}
    
    st.subheader("A Annulus")
    st.session_state.a_config = st.radio("Configuration", ["Long Casing", "Liner"], 
                                        index=0 if st.session_state.a_config == "long_casing" else 1,
                                        key="a_config_radio")
    
    if st.session_state.a_config == "Long Casing":
        create_a_annulus_long_casing_fields()
    else:
        create_a_annulus_liner_fields()

def create_a_annulus_long_casing_fields():
    parameters = [
        ("Safety Valve", [
            ("Collapse Pressure (kPa)", "p_pc_sv", 40000),
            ("Depth (m)", "d_tvd_sv", 1500)
        ]),
        ("Accessory", [
            ("Collapse Pressure (kPa)", "p_pc_acc", 38000),
            ("Depth (m)", "d_tvd_acc", 1400)
        ]),
        ("Packer", [
            ("Collapse Pressure (kPa)", "p_pc_pp", 39000),
            ("Element Rating (kPa)", "p_pkr_pp", 35000),
            ("Depth (m)", "d_tvd_pp", 1600)
        ]),
        ("Tubing", [
            ("Collapse Pressure (kPa)", "p_pc_tbg", 37000)
        ]),
        ("Outer Casing", [
            ("Burst Pressure (kPa)", "p_pb_b", 42000)
        ]),
        ("Rupture Disc", [
            ("Burst Pressure (kPa)", "p_pb_rd", 41000),
            ("Depth (m)", "d_tvd_rd", 1700)
        ]),
        ("Gradients", [
            ("Mud Gradient A-annulus (kPa/m)", "vp_mg_a", 10),
            ("Mud Gradient Tubing (kPa/m)", "vp_mg_tbg", 9.5),
            ("Base Fluid Gradient B-annulus (kPa/m)", "vp_bf_b", 9.7),
            ("Formation Pressure Gradient (kPa/m)", "vp_pform", 10.5)
        ]),
        ("Other Parameters", [
            ("Wellhead Rating (kPa)", "wellhead_rating", 30000),
            ("Annulus Test Pressure (kPa)", "annulus_test_pressure", 25000)
        ])
    ]
    create_parameter_groups(parameters, st.session_state.a_entries, st.session_state.a_checkboxes, "a_")

def create_a_annulus_liner_fields():
    parameters = [
        ("Safety Valve", [
            ("Collapse Pressure (kPa)", "p_pc_sv", 40000),
            ("Depth (m)", "d_tvd_sv", 1500)
        ]),
        ("Accessory", [
            ("Collapse Pressure (kPa)", "p_pc_acc", 38000),
            ("Depth (m)", "d_tvd_acc", 1400)
        ]),
        ("Packer", [
            ("Collapse Pressure (kPa)", "p_pc_pp", 39000),
            ("Element Rating (kPa)", "p_pkr_pp", 35000),
            ("Depth (m)", "d_tvd_pp", 1600)
        ]),
        ("Liner Hanger", [
            ("Pressure Rating (kPa)", "p_lh", 36000),
            ("Burst Pressure (kPa)", "p_pb_lh", 40000),
            ("Depth (m)", "d_tvd_lh", 1800)
        ]),
        ("Tubing", [
            ("Collapse Pressure (kPa)", "p_pc_tbg", 37000)
        ]),
        ("Formation", [
            ("Strength Gradient (kPa/m)", "vp_s_fs_a", 11),
            ("Pressure Gradient (kPa/m)", "vp_form", 10.5),
            ("Shoe Depth (m)", "d_tvd_sh", 2000),
            ("Formation Depth (m)", "d_tvd_form", 1800)
        ]),
        ("Liner Lap", [
            ("Burst Pressure (kPa)", "p_pb_b", 42000)
        ]),
        ("Rupture Disc", [
            ("Burst Pressure (kPa)", "p_pb_rd", 41000),
            ("Depth (m)", "d_tvd_rd", 1700)
        ]),
        ("Gradients", [
            ("Mud Gradient A-annulus (kPa/m)", "vp_mg_a", 10),
            ("Mud Gradient Tubing (kPa/m)", "vp_mg_tbg", 9.5),
            ("Base Fluid Gradient B-annulus (kPa/m)", "vp_bf_b", 9.7),
            ("Formation Pressure Gradient (kPa/m)", "vp_pform", 10.5)
        ]),
        ("Other Parameters", [
            ("Wellhead Rating (kPa)", "wellhead_rating", 30000),
            ("Annulus Test Pressure (kPa)", "annulus_test_pressure", 25000)
        ])
    ]
    create_parameter_groups(parameters, st.session_state.a_entries, st.session_state.a_checkboxes, "a_")

def create_b_annulus_tab():
    if 'b_entries' not in st.session_state:
        st.session_state.b_entries = {}
    if 'b_checkboxes' not in st.session_state:
        st.session_state.b_checkboxes = {}
    
    st.subheader("B Annulus")
    parameters = [
        ("Formation", [
            ("Shoe Depth B-annulus (m)", "d_tvd_sh_b", 2500),
            ("Strength Gradient B-annulus (kPa/m)", "vp_s_fs_b", 11.5)
        ]),
        ("Inner Casing", [
            ("Collapse Pressure (kPa)", "p_pc_b", 43000),
            ("Top of Cement Depth (m)", "d_tvd_toc", 2200)
        ]),
        ("Outer Casing", [
            ("Burst Pressure (kPa)", "p_pb_c", 45000),
            ("Shoe Depth (m)", "d_tvd_sh", 2500)
        ]),
        ("Rupture Disc", [
            ("Burst Pressure (kPa)", "p_pb_rd", 44000),
            ("Depth (m)", "d_tvd_rd", 2300)
        ]),
        ("Gradients", [
            ("Mud Gradient B-annulus (kPa/m)", "vp_mg_b", 10.2),
            ("Mud Gradient A-annulus (kPa/m)", "vp_mg_a", 10),
            ("Base Fluid Gradient C-annulus (kPa/m)", "vp_bf_c", 9.9)
        ]),
        ("Other Parameters", [
            ("Wellhead Rating (kPa)", "wellhead_rating", 30000),
            ("Annulus Test Pressure (kPa)", "annulus_test_pressure", 25000)
        ])
    ]
    create_parameter_groups(parameters, st.session_state.b_entries, st.session_state.b_checkboxes, "b_")

def create_c_annulus_tab():
    if 'c_entries' not in st.session_state:
        st.session_state.c_entries = {}
    if 'c_checkboxes' not in st.session_state:
        st.session_state.c_checkboxes = {}
    
    st.subheader("C Annulus")
    parameters = [
        ("Formation", [
            ("Shoe Depth C-annulus (m)", "d_tvd_sh_c", 3000),
            ("Strength Gradient C-annulus (kPa/m)", "vp_s_fs_c", 12.0)
        ]),
        ("Inner Casing", [
            ("Collapse Pressure (kPa)", "p_pc_c", 48000),
            ("Top of Cement Depth (m)", "d_tvd_toc_c", 2800)
        ]),
        ("Outer Casing", [
            ("Burst Pressure (kPa)", "p_pb_outer_c", 50000),
            ("Shoe Depth (m)", "d_tvd_sh_outer_c", 3000)
        ]),
        ("Rupture Disc", [
            ("Burst Pressure (kPa)", "p_pb_rd_c", 47000),
            ("Depth (m)", "d_tvd_rd_c", 2900)
        ]),
        ("Gradients", [
            ("Mud Gradient C-annulus (kPa/m)", "vp_mg_c", 10.5),
            ("Mud Gradient B-annulus (kPa/m)", "vp_mg_b", 10.2),
            ("Base Fluid Gradient D-annulus (kPa/m)", "vp_bf_d", 10.0)
        ]),
        ("Other Parameters", [
            ("Wellhead Rating (kPa)", "wellhead_rating", 30000),
            ("Annulus Test Pressure (kPa)", "annulus_test_pressure", 25000)
        ])
    ]
    create_parameter_groups(parameters, st.session_state.c_entries, st.session_state.c_checkboxes, "c_")

def create_d_annulus_tab():
    if 'd_entries' not in st.session_state:
        st.session_state.d_entries = {}
    if 'd_checkboxes' not in st.session_state:
        st.session_state.d_checkboxes = {}
    
    st.subheader("D Annulus")
    parameters = [
        ("Formation", [
            ("Shoe Depth D-annulus (m)", "d_tvd_sh_d", 3500),
            ("Strength Gradient D-annulus (kPa/m)", "vp_s_fs_d", 12.5)
        ]),
        ("Inner Casing", [
            ("Collapse Pressure (kPa)", "p_pc_d", 52000),
            ("Top of Cement Depth (m)", "d_tvd_toc_d", 3200)
        ]),
        ("Outer Casing", [
            ("Burst Pressure (kPa)", "p_pb_outer_d", 55000),
            ("Shoe Depth (m)", "d_tvd_sh_outer_d", 3500)
        ]),
        ("Rupture Disc", [
            ("Burst Pressure (kPa)", "p_pb_rd_d", 50000),
            ("Depth (m)", "d_tvd_rd_d", 3300)
        ]),
        ("Gradients", [
            ("Mud Gradient D-annulus (kPa/m)", "vp_mg_d", 10.8),
            ("Mud Gradient C-annulus (kPa/m)", "vp_mg_c", 10.5),
            ("Base Fluid Gradient (kPa/m)", "vp_bf_base", 10.2)
        ]),
        ("Other Parameters", [
            ("Wellhead Rating (kPa)", "wellhead_rating", 30000),
            ("Annulus Test Pressure (kPa)", "annulus_test_pressure", 25000)
        ])
    ]
    create_parameter_groups(parameters, st.session_state.d_entries, st.session_state.d_checkboxes, "d_")

def create_parameter_groups(parameters, entries_dict, checkboxes_dict, annulus_prefix):
    for group_name, params in parameters:
        with st.expander(group_name, expanded=True):
            checkboxes_dict[f"{group_name}_enabled"] = st.checkbox(
                f"Include {group_name} in calculation",
                value=True,
                key=f"{annulus_prefix}{group_name}_enabled"
            )
            col1, col2 = st.columns(2)
            for i, (label, key, default) in enumerate(params):
                with [col1, col2][i % 2]:
                    entries_dict[key] = st.number_input(
                        label,
                        value=float(default),
                        min_value=0.0,
                        step=0.1,
                        format="%.2f",
                        key=f"{annulus_prefix}{group_name}_{key}"
                    )

def create_derating_tab():
    st.header("Derating Factors - ISO 16530-2")
    
    derating_factors = [
        ("Temperature Derating Factor", "temp_factor", 0.95),
        ("Service Life Factor", "service_factor", 0.90),
        ("Manufacturing Factor", "mfg_factor", 0.85),
        ("Design Factor", "design_factor", 0.80),
        ("Installation Factor", "install_factor", 0.95),
        ("Environmental Factor", "env_factor", 0.90)
    ]
    
    st.write("Apply derating factors according to ISO 16530-2")
    for label, key, default in derating_factors:
        st.session_state.derating_entries[key] = st.number_input(
            label,
            value=float(st.session_state.derating_entries.get(key, default)),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.2f",
            key=f"derating_{key}"
        )
    
    st.session_state.apply_derating = st.checkbox("Apply derating factors to results", value=st.session_state.apply_derating, key="apply_derating")

def create_results_tab():
    st.header("MAASP Calculation Results")
    
    if st.session_state.results:
        st.write(f"Well Name: {st.session_state.well_info.get('well_name', '')}")
        st.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write("")
        
        for annulus in ['A_Annulus', 'B_Annulus', 'C_Annulus', 'D_Annulus']:
            if annulus in st.session_state.results:
                st.subheader(annulus.replace('_', ' '))
                st.write(f"MAASP: {st.session_state.results[annulus]:.0f} kPa")
                st.write("Details:")
                for detail in st.session_state.detailed_results.get(annulus, []):
                    st.write(f"  {detail}")
                st.write("")
        
        if st.session_state.apply_derating:
            st.subheader("Applied Derating Factors")
            for label, key in [
                ("Temperature Derating Factor", "temp_factor"),
                ("Service Life Factor", "service_factor"),
                ("Manufacturing Factor", "mfg_factor"),
                ("Design Factor", "design_factor"),
                ("Installation Factor", "install_factor"),
                ("Environmental Factor", "env_factor")
            ]:
                st.write(f"  {label}: {st.session_state.derating_entries[key]}")
            overall_factor = calculate_overall_derating_factor()
            st.write(f"  Overall Derating Factor: {overall_factor:.3f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Export to PDF", key="export_pdf_button"):
                export_to_pdf()
        with col2:
            if st.button("Export to CSV", key="export_csv_button"):
                export_to_csv()
        with col3:
            if st.button("Print Results", key="print_results_button"):
                st.warning("Printing functionality not implemented. Please export to PDF or CSV instead.")

def create_history_tab():
    st.header("Calculation History")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear History", key="clear_history_button"):
            if st.checkbox("Confirm clear history", key="confirm_clear_history"):
                st.session_state.calculation_history = []
                st.success("Calculation history cleared.")
    with col2:
        if st.button("Export History", key="export_history_button"):
            export_history()
    with col3:
        if st.session_state.calculation_history:
            selected_date = st.selectbox("Select calculation to load", 
                                       [calc['date'] for calc in st.session_state.calculation_history],
                                       key="select_calculation")
            if st.button("Load Previous", key="load_previous_button"):
                load_previous_calculation(selected_date)
    
    if st.session_state.calculation_history:
        history_data = [{
            'Date': calc['date'],
            'Well Name': calc['well_info'].get('well_name', ''),
            'A-Annulus MAASP (kPa)': f"{calc['results'].get('A_Annulus', 0):.0f}" if calc['results'].get('A_Annulus') else "-",
            'B-Annulus MAASP (kPa)': f"{calc['results'].get('B_Annulus', 0):.0f}" if calc['results'].get('B_Annulus') else "-",
            'C-Annulus MAASP (kPa)': f"{calc['results'].get('C_Annulus', 0):.0f}" if calc['results'].get('C_Annulus') else "-",
            'D-Annulus MAASP (kPa)': f"{calc['results'].get('D_Annulus', 0):.0f}" if calc['results'].get('D_Annulus') else "-"
        } for calc in st.session_state.calculation_history]
        st.dataframe(pd.DataFrame(history_data))

def calculate_all():
    try:
        results = {}
        detailed_results = {}
        
        if st.session_state.a_config == "Long Casing":
            a_result, a_details = calculate_a_annulus_long_casing()
        else:
            a_result, a_details = calculate_a_annulus_liner()
        
        results['A_Annulus'] = a_result
        detailed_results['A_Annulus'] = a_details
        
        b_result, b_details = calculate_b_annulus()
        results['B_Annulus'] = b_result
        detailed_results['B_Annulus'] = b_details
        
        c_result, c_details = calculate_c_annulus()
        results['C_Annulus'] = c_result
        detailed_results['C_Annulus'] = c_details
        
        d_result, d_details = calculate_d_annulus()
        results['D_Annulus'] = d_result
        detailed_results['D_Annulus'] = d_details
        
        if st.session_state.apply_derating:
            derating_factor = calculate_overall_derating_factor()
            for annulus in results:
                results[annulus] *= derating_factor
        
        st.session_state.results = results
        st.session_state.detailed_results = detailed_results
        
        add_to_history(results)
        st.success("MAASP calculations completed successfully!")
        
        create_results_tab()
        
    except ValueError as e:
        st.error(f"Please check your input values: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred during calculation: {str(e)}")

def calculate_a_annulus_long_casing():
    try:
        results = {}
        details = []
        
        p_pc_sv = float(st.session_state.a_entries.get('p_pc_sv', 0))
        d_tvd_sv = float(st.session_state.a_entries.get('d_tvd_sv', 0))
        vp_mg_a = float(st.session_state.a_entries.get('vp_mg_a', 0))
        vp_mg_tbg = float(st.session_state.a_entries.get('vp_mg_tbg', 0))
        
        if st.session_state.a_checkboxes.get('Safety Valve_enabled', True):
            maasp_sv = p_pc_sv - (d_tvd_sv * (vp_mg_a - vp_mg_tbg))
            results['Safety Valve Collapse'] = maasp_sv
            details.append(f"Safety Valve Collapse: {maasp_sv:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Accessory_enabled', True):
            p_pc_acc = float(st.session_state.a_entries.get('p_pc_acc', 0))
            d_tvd_acc = float(st.session_state.a_entries.get('d_tvd_acc', 0))
            maasp_acc = p_pc_acc - (d_tvd_acc * (vp_mg_a - vp_mg_tbg))
            results['Accessory Collapse'] = maasp_acc
            details.append(f"Accessory Collapse: {maasp_acc:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Packer_enabled', True):
            p_pc_pp = float(st.session_state.a_entries.get('p_pc_pp', 0))
            d_tvd_pp = float(st.session_state.a_entries.get('d_tvd_pp', 0))
            maasp_pp = p_pc_pp - (d_tvd_pp * (vp_mg_a - vp_mg_tbg))
            results['Packer Collapse'] = maasp_pp
            details.append(f"Packer Collapse: {maasp_pp:.0f} kPa")
            
            p_pkr_pp = float(st.session_state.a_entries.get('p_pkr_pp', 0))
            vp_pform = float(st.session_state.a_entries.get('vp_pform', 0))
            maasp_pkr = p_pkr_pp + (d_tvd_pp * (vp_pform - vp_mg_a))
            results['Packer Element Rating'] = maasp_pkr
            details.append(f"Packer Element Rating: {maasp_pkr:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Tubing_enabled', True):
            p_pc_tbg = float(st.session_state.a_entries.get('p_pc_tbg', 0))
            maasp_tbg = p_pc_tbg - (d_tvd_pp * (vp_mg_a - vp_mg_tbg))
            results['Tubing Collapse'] = maasp_tbg
            details.append(f"Tubing Collapse: {maasp_tbg:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Outer Casing_enabled', True):
            p_pb_b = float(st.session_state.a_entries.get('p_pb_b', 0))
            d_tvd_lh = float(st.session_state.a_entries.get('d_tvd_lh', 0))
            vp_bf_b = float(st.session_state.a_entries.get('vp_bf_b', 0))
            maasp_pb = p_pb_b - (d_tvd_lh * (vp_mg_a - vp_bf_b))
            results['Outer Casing Burst'] = maasp_pb
            details.append(f"Outer Casing Burst: {maasp_pb:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Other Parameters_enabled', True):
            wellhead_rating = float(st.session_state.a_entries.get('wellhead_rating', 0))
            results['Wellhead Rating'] = wellhead_rating
            details.append(f"Wellhead Rating: {wellhead_rating:.0f} kPa")
            
            annulus_test_pressure = float(st.session_state.a_entries.get('annulus_test_pressure', 0))
            results['Annulus Test Pressure'] = annulus_test_pressure
            details.append(f"Annulus Test Pressure: {annulus_test_pressure:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Rupture Disc_enabled', True):
            p_pb_rd = float(st.session_state.a_entries.get('p_pb_rd', 41000))
            d_tvd_rd = float(st.session_state.a_entries.get('d_tvd_rd', 1700))
            vp_bf_b = float(st.session_state.a_entries.get('vp_bf_b', 9.7))
            maasp_rd = p_pb_rd - (d_tvd_rd * (vp_mg_a - vp_bf_b))
            results['Rupture Disc'] = maasp_rd
            details.append(f"Rupture Disc: {maasp_rd:.0f} kPa")
        
        if results:
            min_maasp = min(results.values())
            controlling_factor = min(results, key=results.get)
            details.append(f"\nControlling Factor: {controlling_factor}")
            details.append(f"Minimum MAASP: {min_maasp:.0f} kPa")
            return min_maasp, details
        else:
            return 0, ["No parameters enabled for calculation"]
    except Exception as e:
        return 0, [f"Error in A-annulus long casing calculation: {str(e)}"]

def calculate_a_annulus_liner():
    try:
        results = {}
        details = []
        
        vp_mg_a = float(st.session_state.a_entries.get('vp_mg_a', 0))
        vp_mg_tbg = float(st.session_state.a_entries.get('vp_mg_tbg', 0))
        vp_pform = float(st.session_state.a_entries.get('vp_pform', 0))
        vp_bf_b = float(st.session_state.a_entries.get('vp_bf_b', 0))
        
        if st.session_state.a_checkboxes.get('Safety Valve_enabled', True):
            p_pc_sv = float(st.session_state.a_entries.get('p_pc_sv', 0))
            d_tvd_sv = float(st.session_state.a_entries.get('d_tvd_sv', 0))
            maasp_sv = p_pc_sv - (d_tvd_sv * (vp_mg_a - vp_mg_tbg))
            results['Safety Valve Collapse'] = maasp_sv
            details.append(f"Safety Valve Collapse: {maasp_sv:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Accessory_enabled', True):
            p_pc_acc = float(st.session_state.a_entries.get('p_pc_acc', 0))
            d_tvd_acc = float(st.session_state.a_entries.get('d_tvd_acc', 0))
            maasp_acc = p_pc_acc - (d_tvd_acc * (vp_mg_a - vp_mg_tbg))
            results['Accessory Collapse'] = maasp_acc
            details.append(f"Accessory Collapse: {maasp_acc:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Packer_enabled', True):
            p_pc_pp = float(st.session_state.a_entries.get('p_pc_pp', 0))
            d_tvd_pp = float(st.session_state.a_entries.get('d_tvd_pp', 0))
            maasp_pp = p_pc_pp - (d_tvd_pp * (vp_mg_a - vp_mg_tbg))
            results['Packer Collapse'] = maasp_pp
            details.append(f"Packer Collapse: {maasp_pp:.0f} kPa")
            
            p_pkr_pp = float(st.session_state.a_entries.get('p_pkr_pp', 0))
            maasp_pkr = p_pkr_pp + (d_tvd_pp * (vp_pform - vp_mg_a))
            results['Packer Element Rating'] = maasp_pkr
            details.append(f"Packer Element Rating: {maasp_pkr:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Liner Hanger_enabled', True):
            p_lh = float(st.session_state.a_entries.get('p_lh', 0))
            d_tvd_lh = float(st.session_state.a_entries.get('d_tvd_lh', 0))
            vp_form = float(st.session_state.a_entries.get('vp_form', 0))
            d_tvd_form = float(st.session_state.a_entries.get('d_tvd_form', 0))
            
            maasp_lh = p_lh + (d_tvd_lh * (vp_pform - vp_mg_a)) - (vp_form * (d_tvd_form - d_tvd_lh))
            results['Liner Hanger Rating'] = maasp_lh
            details.append(f"Liner Hanger Rating: {maasp_lh:.0f} kPa")
            
            p_pb_lh = float(st.session_state.a_entries.get('p_pb_lh', 0))
            maasp_lh_burst = p_pb_lh - (d_tvd_lh * (vp_mg_a - vp_bf_b))
            results['Liner Hanger Burst'] = maasp_lh_burst
            details.append(f"Liner Hanger Burst: {maasp_lh_burst:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Tubing_enabled', True):
            p_pc_tbg = float(st.session_state.a_entries.get('p_pc_tbg', 0))
            d_tvd_pp = float(st.session_state.a_entries.get('d_tvd_pp', 0))
            maasp_tbg = p_pc_tbg - (d_tvd_pp * (vp_mg_a - vp_mg_tbg))
            results['Tubing Collapse'] = maasp_tbg
            details.append(f"Tubing Collapse: {maasp_tbg:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Formation_enabled', True):
            d_tvd_sh = float(st.session_state.a_entries.get('d_tvd_sh', 0))
            vp_s_fs_a = float(st.session_state.a_entries.get('vp_s_fs_a', 0))
            maasp_fs = d_tvd_sh * (vp_s_fs_a - vp_mg_a)
            results['Formation Strength'] = maasp_fs
            details.append(f"Formation Strength: {maasp_fs:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Liner Lap_enabled', True):
            p_pb_b = float(st.session_state.a_entries.get('p_pb_b', 0))
            d_tvd_pp = float(st.session_state.a_entries.get('d_tvd_pp', 0))
            maasp_pb = p_pb_b - (d_tvd_pp * (vp_mg_a - vp_bf_b))
            results['Liner Lap Burst'] = maasp_pb
            details.append(f"Liner Lap Burst: {maasp_pb:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Other Parameters_enabled', True):
            wellhead_rating = float(st.session_state.a_entries.get('wellhead_rating', 0))
            results['Wellhead Rating'] = wellhead_rating
            details.append(f"Wellhead Rating: {wellhead_rating:.0f} kPa")
            
            annulus_test_pressure = float(st.session_state.a_entries.get('annulus_test_pressure', 0))
            results['Annulus Test Pressure'] = annulus_test_pressure
            details.append(f"Annulus Test Pressure: {annulus_test_pressure:.0f} kPa")
        
        if st.session_state.a_checkboxes.get('Rupture Disc_enabled', True):
            p_pb_rd = float(st.session_state.a_entries.get('p_pb_rd', 0))
            d_tvd_rd = float(st.session_state.a_entries.get('d_tvd_rd', 0))
            maasp_rd = p_pb_rd - (d_tvd_rd * (vp_mg_a - vp_bf_b))
            results['Rupture Disc'] = maasp_rd
            details.append(f"Rupture Disc: {maasp_rd:.0f} kPa")
        
        if results:
            min_maasp = min(results.values())
            controlling_factor = min(results, key=results.get)
            details.append(f"\nControlling Factor: {controlling_factor}")
            details.append(f"Minimum MAASP: {min_maasp:.0f} kPa")
            return min_maasp, details
        else:
            return 0, ["No parameters enabled for calculation"]
    except Exception as e:
        return 0, [f"Error in A-annulus liner calculation: {str(e)}"]

def calculate_b_annulus():
    try:
        results = {}
        details = []
        
        vp_mg_b = float(st.session_state.b_entries.get('vp_mg_b', 0))
        vp_mg_a = float(st.session_state.b_entries.get('vp_mg_a', 0))
        vp_bf_c = float(st.session_state.b_entries.get('vp_bf_c', 0))
        
        if st.session_state.b_checkboxes.get('Formation_enabled', True):
            d_tvd_sh_b = float(st.session_state.b_entries.get('d_tvd_sh_b', 0))
            vp_s_fs_b = float(st.session_state.b_entries.get('vp_s_fs_b', 0))
            maasp_fs = d_tvd_sh_b * (vp_s_fs_b - vp_mg_b)
            results['Formation Strength'] = maasp_fs
            details.append(f"Formation Strength: {maasp_fs:.0f} kPa")
        
        if st.session_state.b_checkboxes.get('Inner Casing_enabled', True):
            p_pc_b = float(st.session_state.b_entries.get('p_pc_b', 0))
            d_tvd_toc = float(st.session_state.b_entries.get('d_tvd_toc', 0))
            maasp_pc = p_pc_b - (d_tvd_toc * (vp_mg_b - vp_mg_a))
            results['Inner Casing Collapse'] = maasp_pc
            details.append(f"Inner Casing Collapse: {maasp_pc:.0f} kPa")
        
        if st.session_state.b_checkboxes.get('Outer Casing_enabled', True):
            p_pb_c = float(st.session_state.b_entries.get('p_pb_c', 0))
            d_tvd_sh = float(st.session_state.b_entries.get('d_tvd_sh', 0))
            maasp_pb = p_pb_c - (d_tvd_sh * (vp_mg_b - vp_bf_c))
            results['Outer Casing Burst'] = maasp_pb
            details.append(f"Outer Casing Burst: {maasp_pb:.0f} kPa")
        
        if st.session_state.b_checkboxes.get('Other Parameters_enabled', True):
            wellhead_rating = float(st.session_state.b_entries.get('wellhead_rating', 0))
            results['Wellhead Rating'] = wellhead_rating
            details.append(f"Wellhead Rating: {wellhead_rating:.0f} kPa")
            
            annulus_test_pressure = float(st.session_state.b_entries.get('annulus_test_pressure', 0))
            results['Annulus Test Pressure'] = annulus_test_pressure
            details.append(f"Annulus Test Pressure: {annulus_test_pressure:.0f} kPa")
        
        if st.session_state.b_checkboxes.get('Rupture Disc_enabled', True):
            p_pb_rd = float(st.session_state.b_entries.get('p_pb_rd', 0))
            d_tvd_rd = float(st.session_state.b_entries.get('d_tvd_rd', 0))
            maasp_rd = p_pb_rd - (d_tvd_rd * (vp_mg_b - vp_bf_c))
            results['Rupture Disc'] = maasp_rd
            details.append(f"Rupture Disc: {maasp_rd:.0f} kPa")
        
        if results:
            min_maasp = min(results.values())
            controlling_factor = min(results, key=results.get)
            details.append(f"\nControlling Factor: {controlling_factor}")
            details.append(f"Minimum MAASP: {min_maasp:.0f} kPa")
            return min_maasp, details
        else:
            return 0, ["No parameters enabled for calculation"]
    except Exception as e:
        return 0, [f"Error in B-annulus calculation: {str(e)}"]

def calculate_c_annulus():
    try:
        results = {}
        details = []
        
        vp_mg_c = float(st.session_state.c_entries.get('vp_mg_c', 0))
        vp_mg_b = float(st.session_state.c_entries.get('vp_mg_b', 0))
        vp_bf_d = float(st.session_state.c_entries.get('vp_bf_d', 0))
        
        if st.session_state.c_checkboxes.get('Formation_enabled', True):
            d_tvd_sh_c = float(st.session_state.c_entries.get('d_tvd_sh_c', 0))
            vp_s_fs_c = float(st.session_state.c_entries.get('vp_s_fs_c', 0))
            maasp_fs = d_tvd_sh_c * (vp_s_fs_c - vp_mg_c)
            results['Formation Strength'] = maasp_fs
            details.append(f"Formation Strength: {maasp_fs:.0f} kPa")
        
        if st.session_state.c_checkboxes.get('Inner Casing_enabled', True):
            p_pc_c = float(st.session_state.c_entries.get('p_pc_c', 0))
            d_tvd_toc_c = float(st.session_state.c_entries.get('d_tvd_toc_c', 0))
            maasp_pc = p_pc_c - (d_tvd_toc_c * (vp_mg_c - vp_mg_b))
            results['Inner Casing Collapse'] = maasp_pc
            details.append(f"Inner Casing Collapse: {maasp_pc:.0f} kPa")
        
        if st.session_state.c_checkboxes.get('Outer Casing_enabled', True):
            p_pb_outer_c = float(st.session_state.c_entries.get('p_pb_outer_c', 0))
            d_tvd_sh_outer_c = float(st.session_state.c_entries.get('d_tvd_sh_outer_c', 0))
            maasp_pb = p_pb_outer_c - (d_tvd_sh_outer_c * (vp_mg_c - vp_bf_d))
            results['Outer Casing Burst'] = maasp_pb
            details.append(f"Outer Casing Burst: {maasp_pb:.0f} kPa")
        
        if st.session_state.c_checkboxes.get('Other Parameters_enabled', True):
            wellhead_rating = float(st.session_state.c_entries.get('wellhead_rating', 0))
            results['Wellhead Rating'] = wellhead_rating
            details.append(f"Wellhead Rating: {wellhead_rating:.0f} kPa")
            
            annulus_test_pressure = float(st.session_state.c_entries.get('annulus_test_pressure', 0))
            results['Annulus Test Pressure'] = annulus_test_pressure
            details.append(f"Annulus Test Pressure: {annulus_test_pressure:.0f} kPa")
        
        if st.session_state.c_checkboxes.get('Rupture Disc_enabled', True):
            p_pb_rd_c = float(st.session_state.c_entries.get('p_pb_rd_c', 0))
            d_tvd_rd_c = float(st.session_state.c_entries.get('d_tvd_rd_c', 0))
            maasp_rd = p_pb_rd_c - (d_tvd_rd_c * (vp_mg_c - vp_bf_d))
            results['Rupture Disc'] = maasp_rd
            details.append(f"Rupture Disc: {maasp_rd:.0f} kPa")
        
        if results:
            min_maasp = min(results.values())
            controlling_factor = min(results, key=results.get)
            details.append(f"\nControlling Factor: {controlling_factor}")
            details.append(f"Minimum MAASP: {min_maasp:.0f} kPa")
            return min_maasp, details
        else:
            return 0, ["No parameters enabled for calculation"]
    except Exception as e:
        return 0, [f"Error in C-annulus calculation: {str(e)}"]

def calculate_d_annulus():
    try:
        results = {}
        details = []
        
        vp_mg_d = float(st.session_state.d_entries.get('vp_mg_d', 0))
        vp_mg_c = float(st.session_state.d_entries.get('vp_mg_c', 0))
        vp_bf_base = float(st.session_state.d_entries.get('vp_bf_base', 0))
        
        if st.session_state.d_checkboxes.get('Formation_enabled', True):
            d_tvd_sh_d = float(st.session_state.d_entries.get('d_tvd_sh_d', 0))
            vp_s_fs_d = float(st.session_state.d_entries.get('vp_s_fs_d', 0))
            maasp_fs = d_tvd_sh_d * (vp_s_fs_d - vp_mg_d)
            results['Formation Strength'] = maasp_fs
            details.append(f"Formation Strength: {maasp_fs:.0f} kPa")
        
        if st.session_state.d_checkboxes.get('Inner Casing_enabled', True):
            p_pc_d = float(st.session_state.d_entries.get('p_pc_d', 0))
            d_tvd_toc_d = float(st.session_state.d_entries.get('d_tvd_toc_d', 0))
            maasp_pc = p_pc_d - (d_tvd_toc_d * (vp_mg_d - vp_mg_c))
            results['Inner Casing Collapse'] = maasp_pc
            details.append(f"Inner Casing Collapse: {maasp_pc:.0f} kPa")
        
        if st.session_state.d_checkboxes.get('Outer Casing_enabled', True):
            p_pb_outer_d = float(st.session_state.d_entries.get('p_pb_outer_d', 0))
            d_tvd_sh_outer_d = float(st.session_state.d_entries.get('d_tvd_sh_outer_d', 0))
            maasp_pb = p_pb_outer_d - (d_tvd_sh_outer_d * (vp_mg_d - vp_bf_base))
            results['Outer Casing Burst'] = maasp_pb
            details.append(f"Outer Casing Burst: {maasp_pb:.0f} kPa")
        
        if st.session_state.d_checkboxes.get('Other Parameters_enabled', True):
            wellhead_rating = float(st.session_state.d_entries.get('wellhead_rating', 0))
            results['Wellhead Rating'] = wellhead_rating
            details.append(f"Wellhead Rating: {wellhead_rating:.0f} kPa")
            
            annulus_test_pressure = float(st.session_state.d_entries.get('annulus_test_pressure', 0))
            results['Annulus Test Pressure'] = annulus_test_pressure
            details.append(f"Annulus Test Pressure: {annulus_test_pressure:.0f} kPa")
        
        if st.session_state.d_checkboxes.get('Rupture Disc_enabled', True):
            p_pb_rd_d = float(st.session_state.d_entries.get('p_pb_rd_d', 0))
            d_tvd_rd_d = float(st.session_state.d_entries.get('d_tvd_rd_d', 0))
            maasp_rd = p_pb_rd_d - (d_tvd_rd_d * (vp_mg_d - vp_bf_base))
            results['Rupture Disc'] = maasp_rd
            details.append(f"Rupture Disc: {maasp_rd:.0f} kPa")
        
        if results:
            min_maasp = min(results.values())
            controlling_factor = min(results, key=results.get)
            details.append(f"\nControlling Factor: {controlling_factor}")
            details.append(f"Minimum MAASP: {min_maasp:.0f} kPa")
            return min_maasp, details
        else:
            return 0, ["No parameters enabled for calculation"]
    except Exception as e:
        return 0, [f"Error in D-annulus calculation: {str(e)}"]

def calculate_overall_derating_factor():
    try:
        temp_factor = float(st.session_state.derating_entries.get('temp_factor', 1.0))
        service_factor = float(st.session_state.derating_entries.get('service_factor', 1.0))
        mfg_factor = float(st.session_state.derating_entries.get('mfg_factor', 1.0))
        design_factor = float(st.session_state.derating_entries.get('design_factor', 1.0))
        install_factor = float(st.session_state.derating_entries.get('install_factor', 1.0))
        env_factor = float(st.session_state.derating_entries.get('env_factor', 1.0))
        
        overall_factor = temp_factor * service_factor * mfg_factor * design_factor * install_factor * env_factor
        return overall_factor
    except ValueError as e:
        st.error(f"Invalid derating factor value: {str(e)}")
        return 1.0
    except Exception as e:
        st.error(f"Error calculating derating factor: {str(e)}")
        return 1.0

def add_to_history(results):
    calc_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.calculation_history.append({
        'date': calc_date,
        'well_info': st.session_state.well_info,
        'results': results,
        'detailed_results': st.session_state.detailed_results,
        'a_config': st.session_state.a_config,
        'a_entries': st.session_state.a_entries,
        'b_entries': st.session_state.b_entries,
        'c_entries': st.session_state.c_entries,
        'd_entries': st.session_state.d_entries,
        'derating_entries': st.session_state.derating_entries,
        'apply_derating': st.session_state.apply_derating
    })

def clear_all():
    if st.checkbox("Confirm clear all inputs and results", key="confirm_clear_all"):
        st.session_state.well_info = {key: "" for key in st.session_state.well_info}
        st.session_state.well_info['date'] = datetime.datetime.now().strftime("%Y-%m-%d")
        st.session_state.a_entries = {}
        st.session_state.b_entries = {}
        st.session_state.c_entries = {}
        st.session_state.d_entries = {}
        st.session_state.derating_entries = {
            'temp_factor': 0.95,
            'service_factor': 0.90,
            'mfg_factor': 0.85,
            'design_factor': 0.80,
            'install_factor': 0.95,
            'env_factor': 0.90
        }
        st.session_state.apply_derating = True
        st.session_state.results = {}
        st.session_state.detailed_results = {}
        st.success("All inputs and results cleared.")
        st.rerun()

def save_config():
    config = {
        'well_info': st.session_state.well_info,
        'a_config': st.session_state.a_config,
        'a_entries': st.session_state.a_entries,
        'b_entries': st.session_state.b_entries,
        'c_entries': st.session_state.c_entries,
        'd_entries': st.session_state.d_entries,
        'derating_entries': st.session_state.derating_entries,
        'apply_derating': st.session_state.apply_derating
    }
    
    buffer = io.StringIO()
    json.dump(config, buffer, indent=4)
    buffer.seek(0)
    st.download_button(
        label="Save Configuration",
        data=buffer,
        file_name="maasp_config.json",
        mime="application/json",
        key="save_config_button"
    )

def load_config():
    uploaded_file = st.file_uploader("Load Configuration", type=["json"], key="load_config_uploader")
    if uploaded_file:
        try:
            config = json.load(uploaded_file)
            
            for key, value in config.get('well_info', {}).items():
                st.session_state.well_info[key] = value
            
            st.session_state.a_config = config.get('a_config', 'long_casing')
            
            for key, value in config.get('a_entries', {}).items():
                st.session_state.a_entries[key] = value
            
            for entries, config_key in [
                (st.session_state.b_entries, 'b_entries'),
                (st.session_state.c_entries, 'c_entries'),
                (st.session_state.d_entries, 'd_entries')
            ]:
                for key, value in config.get(config_key, {}).items():
                    entries[key] = value
            
            for key, value in config.get('derating_entries', {}).items():
                st.session_state.derating_entries[key] = value
            
            st.session_state.apply_derating = config.get('apply_derating', True)
            
            st.success("Configuration loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load configuration: {str(e)}")

def export_to_pdf():
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("MAASP Calculation Results", styles['Title']))
        story.append(Spacer(1, 12))
        
        well_name = st.session_state.well_info.get('well_name', '')
        calc_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Well Name: {well_name}", styles['Normal']))
        story.append(Paragraph(f"Date: {calc_date}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        for annulus in ['A_Annulus', 'B_Annulus', 'C_Annulus', 'D_Annulus']:
            if annulus in st.session_state.results:
                story.append(Paragraph(f"{annulus.replace('_', ' ')}:", styles['Heading2']))
                story.append(Paragraph(f"MAASP: {st.session_state.results[annulus]:.0f} kPa", styles['Normal']))
                story.append(Paragraph("Details:", styles['Normal']))
                for detail in st.session_state.detailed_results.get(annulus, []):
                    story.append(Paragraph(f"  {detail}", styles['Normal']))
                story.append(Spacer(1, 12))
        
        if st.session_state.apply_derating:
            story.append(Paragraph("Applied Derating Factors:", styles['Heading2']))
            for label, key in [
                ("Temperature Derating Factor", "temp_factor"),
                ("Service Life Factor", "service_factor"),
                ("Manufacturing Factor", "mfg_factor"),
                ("Design Factor", "design_factor"),
                ("Installation Factor", "install_factor"),
                ("Environmental Factor", "env_factor")
            ]:
                story.append(Paragraph(f"  {label}: {st.session_state.derating_entries[key]}", styles['Normal']))
            overall_factor = calculate_overall_derating_factor()
            story.append(Paragraph(f"  Overall Derating Factor: {overall_factor:.3f}", styles['Normal']))
        
        doc.build(story)
        st.markdown(create_download_link(buffer, "maasp_results.pdf", "Download PDF"), unsafe_allow_html=True)
    except ImportError:
        st.error("ReportLab library not installed. Please install it to export to PDF.")
    except Exception as e:
        st.error(f"Failed to export to PDF: {str(e)}")

def export_to_csv():
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    
    writer.writerow(["Well Information"])
    for key, value in st.session_state.well_info.items():
        writer.writerow([key.replace('_', ' ').title(), value])
    writer.writerow([])
    
    writer.writerow(["Annulus Results"])
    writer.writerow(["Annulus", "MAASP (kPa)", "Details"])
    for annulus in ['A_Annulus', 'B_Annulus', 'C_Annulus', 'D_Annulus']:
        if annulus in st.session_state.results:
            details = "; ".join(st.session_state.detailed_results.get(annulus, []))
            writer.writerow([annulus.replace('_', ' '), f"{st.session_state.results[annulus]:.0f}", details])
    writer.writerow([])
    
    if st.session_state.apply_derating:
        writer.writerow(["Derating Factors"])
        for label, key in [
            ("Temperature Derating Factor", "temp_factor"),
            ("Service Life Factor", "service_factor"),
            ("Manufacturing Factor", "mfg_factor"),
            ("Design Factor", "design_factor"),
            ("Installation Factor", "install_factor"),
            ("Environmental Factor", "env_factor")
        ]:
            writer.writerow([label, st.session_state.derating_entries[key]])
        overall_factor = calculate_overall_derating_factor()
        writer.writerow(["Overall Derating Factor", f"{overall_factor:.3f}"])
    
    buffer.seek(0)
    st.markdown(create_download_link(buffer, "maasp_results.csv", "Download CSV"), unsafe_allow_html=True)

def export_history():
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['Date', 'Well Name', 'A-Annulus MAASP (kPa)', 'B-Annulus MAASP (kPa)', 'C-Annulus MAASP (kPa)', 'D-Annulus MAASP (kPa)'])
    
    for calc in st.session_state.calculation_history:
        a_maasp = calc['results'].get('A_Annulus', 0)
        b_maasp = calc['results'].get('B_Annulus', 0)
        c_maasp = calc['results'].get('C_Annulus', 0)
        d_maasp = calc['results'].get('D_Annulus', 0)
        writer.writerow([
            calc['date'],
            calc['well_info'].get('well_name', ''),
            f"{a_maasp:.0f}" if a_maasp else "-",
            f"{b_maasp:.0f}" if b_maasp else "-",
            f"{c_maasp:.0f}" if c_maasp else "-",
            f"{d_maasp:.0f}" if d_maasp else "-"
        ])
    
    buffer.seek(0)
    st.markdown(create_download_link(buffer, "maasp_history.csv", "Download History CSV"), unsafe_allow_html=True)

def load_previous_calculation(date):
    for calc in st.session_state.calculation_history:
        if calc['date'] == date:
            try:
                st.session_state.well_info = calc['well_info']
                st.session_state.a_config = calc['a_config']
                st.session_state.a_entries = calc['a_entries']
                st.session_state.b_entries = calc['b_entries']
                st.session_state.c_entries = calc['c_entries']
                st.session_state.d_entries = calc['d_entries']
                st.session_state.derating_entries = calc['derating_entries']
                st.session_state.apply_derating = calc['apply_derating']
                st.session_state.results = calc['results']
                st.session_state.detailed_results = calc['detailed_results']
                
                st.success("Previous calculation loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load previous calculation: {str(e)}")
            return
    st.error("Selected calculation not found in history.")

if __name__ == "__main__":
    main()
