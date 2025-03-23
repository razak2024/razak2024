import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import io

class WellWashingAnalysis:
    def __init__(self):
        self.df = None
        self.df_sample = self.create_sample_df()

    def create_sample_df(self):
        """Creates a sample dataframe from the provided data"""
        data = {
            'Puits': ['GT-11', 'GT-113', 'GT-113', 'GT-113', 'GT-14', 'GT-14', 'GT-17', 'GT-17', 
                     'GT-19', 'GT-19', 'GT-24', 'GT-25BIS', 'GT-25BIS', 'GT-25BIS', 'GT-25BIS', 
                     'GT-25BIS', 'GT-25BIS', 'GT-25BIS', 'GT-27', 'GT-27', 'GT-39', 'GT-39', 
                     'GT-39C', 'GT-39C', 'GT-39C', 'GT-4', 'GT-4', 'GT-4', 'GT-44', 'GT-44', 
                     'GT-44', 'GT-44', 'GT-44', 'GT-44', 'GT-44', 'GT-44', 'GT-47', 'GT-47', 
                     'GT-47', 'GT-48', 'GT-48', 'GT-50', 'GT-50', 'GT-50', 'GT-51', 'GT-8', 
                     'GT-8', 'HC-4', 'HC-4', 'HC-4', 'HC-5'],
            'Périmètre': ['GTL', 'GTS', 'GTS', 'GTS', 'GTL', 'GTL', 'GTL', 'GTL', 
                         'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 
                         'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 
                         'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 
                         'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 
                         'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 'GTL', 
                         'GTL', 'HCT', 'HCT', 'HCT', 'HCT'],
            'Date_de_Fermeture': ['-', '-', '-', '-', '-', '-', '-', '-', 
                                '09/03/2025', '16/03/2025', '-', '-', '-', '-', '-', 
                                '-', '-', '-', '-', '-', '-', '-', 
                                '-', '-', '-', '-', '-', '-', '-', '-', 
                                '-', '-', '-', '-', '-', '-', '04/03/2025', '08/03/2025', 
                                '19/03/2025', '-', '-', '-', '-', '-', '-', '-', 
                                '-', '20/03/2025', '20/03/2025', '20/03/2025', '04/03/2025'],
            'Heures_d_Arret': [6, 6, 6, 6, 6, 7, 8, 7, 
                              9, 9, 7, 6, 6, 6, 8, 
                              7, 7, 8, 6, 7, 7, 6, 
                              6, 7, 6, 6, 8, 7, 6, 6, 
                              7, 6, 6, 7, 9, 8, 10, 9, 
                              8, 8, 6, 6, 7, 7, 2, 6, 
                              8, 12, 10, 24, 5],
            'MAP': [2.612, 3.441, 3.944, 4.211, 3.052, 4.022, 8.933, 8.523, 
                   2.004, 2.227, 4.237, 6.46, 6.8, 7.144, 7.79, 
                   7.067, 6.911, 7.967, 2.564, 3.423, 6.704, 6.479, 
                   1.178, 1.524, 1.352, 10.198, 12.942, 11.602, 4.657, 4.671, 
                   5.729, 5.348, 5.094, 5.508, 6.817, 5.71, 5.873, 5.291, 
                   4.368, 12.474, 9.449, 9.753, 12.166, 12.193, 1.307, 4.151, 
                   3.727, 2.168, 1.569, 3.765, 3.263],
            'Cause_d_Arret': ['LAVAGE', 'LAVAGE', 'LAVAGE', 'GTL_FERME POUR LAVAGE', 'LAVAGE', 'LAVAGE', 'LAVAGE', 'GTL_FERME POUR LAVAGE', 
                             'LAVAGE', 'LAVAGE', 'LAVAGE', 'LAVAGE', 'LAVAGE', 'LAVAGE', 'LAVAGE', 
                             'LAVAGE', 'GTL_FERME POUR LAVAGE', 'GTL_FERME POUR LAVAGE', 'LAVAGE', 'GTL_FERME POUR LAVAGE', 'LAVAGE', 'LAVAGE', 
                             'LAVAGE', 'LAVAGE', 'GTL_FERME POUR LAVAGE', 'LAVAGE', 'LAVAGE', 'GTL_FERME POUR LAVAGE', 'LAVAGE', 'LAVAGE', 
                             'LAVAGE', 'LAVAGE', 'LAVAGE', 'LAVAGE', 'GTL_FERME POUR LAVAGE', 'GTL_FERME POUR LAVAGE', 'LAVAGE', 'LAVAGE', 
                             'GTL_FERME POUR LAVAGE', 'LAVAGE', 'LAVAGE', 'LAVAGE', 'LAVAGE', 'GTL_FERME POUR LAVAGE', 'GTL_FERME POUR LAVAGE', 'LAVAGE', 
                             'LAVAGE', 'GTL_FERME POUR LAVAGE', 'GTL_FERME POUR LAVAGE', 'GTL_FERME POUR LAVAGE', 'LAVAGE']
        }
        
        df = pd.DataFrame(data)
        
        # Convert dates
        df['Date_de_Fermeture'] = pd.to_datetime(df['Date_de_Fermeture'], format='%d/%m/%Y', errors='coerce')
        
        return df

    def optimize_downtime_hours(self):
        """Optimizes the reduction of downtime hours"""
        # Analyze downtime hours by well
        hours_by_well = self.df.groupby('Puits')['Heures_d_Arret'].agg(['mean', 'min', 'max', 'sum', 'count']).reset_index()
        hours_by_well.columns = ['Puits', 'Moyenne', 'Minimum', 'Maximum', 'Total', 'Nombre de lavages']
        
        # Find wells that could benefit from a reduction in hours
        hours_by_well['Potentiel de réduction'] = hours_by_well['Moyenne'] - hours_by_well['Minimum']
        hours_by_well['Économie potentielle (heures)'] = hours_by_well['Potentiel de réduction'] * hours_by_well['Nombre de lavages']
        
        # Sort by savings potential
        hours_by_well = hours_by_well.sort_values('Économie potentielle (heures)', ascending=False)
        
        return hours_by_well

    def optimize_map_per_hour(self):
        """Optimizes MAP per downtime hour"""
        # Calculate efficiency (MAP per hour) for each washing
        self.df['Efficacité'] = self.df['MAP'] / self.df['Heures_d_Arret']
        
        # Calculate average efficiency per well
        efficiency_by_well = self.df.groupby('Puits')['Efficacité'].agg(['mean', 'min', 'max', 'count']).reset_index()
        efficiency_by_well.columns = ['Puits', 'Efficacité moyenne', 'Efficacité min', 'Efficacité max', 'Nombre de lavages']
        
        # Calculate average MAP per well
        map_by_well = self.df.groupby('Puits')['MAP'].mean().reset_index()
        map_by_well.columns = ['Puits', 'MAP moyen']
        
        # Merge dataframes
        analysis = pd.merge(efficiency_by_well, map_by_well, on='Puits')
        
        # Calculate the gap between max and average
        analysis['Potentiel d\'amélioration'] = analysis['Efficacité max'] - analysis['Efficacité moyenne']
        analysis['Potentiel en %'] = (analysis['Potentiel d\'amélioration'] / analysis['Efficacité moyenne']) * 100
        
        # Sort by improvement potential
        analysis = analysis.sort_values('Potentiel en %', ascending=False)
        
        return analysis

    def plan_washings(self):
        """Plans washings optimally"""
        if 'Date_de_Fermeture' not in self.df.columns or self.df['Date_de_Fermeture'].isna().all():
            return None, "Pas assez de données de dates pour planifier les lavages"
        
        # Filter data with valid dates
        df_dates = self.df[self.df['Date_de_Fermeture'].notna()].copy()
        
        if len(df_dates) < 2:
            return None, "Données de dates insuffisantes pour une planification fiable"
        
        # Check if there are enough dates to calculate frequency
        wells_multiple_washings = df_dates['Puits'].value_counts()[df_dates['Puits'].value_counts() > 1].index.tolist()
        
        if not wells_multiple_washings:
            return None, "Pas assez de données historiques pour établir une fréquence de lavage"
        
        # For each well with multiple dated washings, calculate the average frequency
        frequencies = []
        well_data = []
        
        for well in wells_multiple_washings:
            dates = df_dates[df_dates['Puits'] == well]['Date_de_Fermeture'].sort_values().reset_index(drop=True)
            
            if len(dates) > 1:
                # Calculate intervals in days
                intervals = []
                for i in range(1, len(dates)):
                    interval = (dates[i] - dates[i-1]).days
                    intervals.append(interval)
                
                mean_interval = np.mean(intervals)
                frequencies.append({'Puits': well, 'Intervalle moyen (jours)': mean_interval})
                
                well_info = {
                    'well': well,
                    'mean_interval': mean_interval,
                    'dates': [d.strftime('%d/%m/%Y') for d in dates]
                }
                
                if len(intervals) > 1:
                    std_interval = np.std(intervals)
                    well_info['std_interval'] = std_interval
                else:
                    well_info['std_interval'] = None
                    
                well_data.append(well_info)
        
        # Create a dataframe of frequencies
        if frequencies:
            df_frequencies = pd.DataFrame(frequencies)
            
            # Plan the next washings
            today = datetime.now()
            
            planned_dates = []
            for _, row in df_frequencies.iterrows():
                well = row['Puits']
                interval = row['Intervalle moyen (jours)']
                
                # Find the date of the last washing
                last_date = df_dates[df_dates['Puits'] == well]['Date_de_Fermeture'].max()
                
                # Calculate the next washing date
                days_since_last = (today - last_date).days
                days_before_next = max(0, round(interval - days_since_last))
                next_date = today + pd.Timedelta(days=days_before_next)
                
                # Evaluate priority
                if days_since_last > interval:
                    priority = "HAUTE - Retard de {} jours".format(days_since_last - int(interval))
                elif days_since_last > interval * 0.8:
                    priority = "MOYENNE - Approche de l'échéance"
                else:
                    priority = "BASSE - Dans les délais"
                
                planned_dates.append({
                    'Puits': well,
                    'Dernier_lavage': last_date,
                    'Prochain_lavage': next_date,
                    'Jours_avant': days_before_next,
                    'Priorité': priority
                })
            
            # Create a dataframe of planned dates
            df_planned = pd.DataFrame(planned_dates)
            df_planned = df_planned.sort_values('Prochain_lavage')
            
            # Identify washings close in time (less than 2 days apart)
            close_washings = []
            for i in range(1, len(df_planned)):
                gap = (df_planned.iloc[i]['Prochain_lavage'] - df_planned.iloc[i-1]['Prochain_lavage']).days
                if gap < 2:
                    close_washings.append((df_planned.iloc[i-1]['Puits'], df_planned.iloc[i]['Puits']))
            
            return {
                'frequency_data': well_data,
                'planned_dates': df_planned,
                'close_washings': close_washings
            }, None
        
        return None, "Impossible de calculer les fréquences de lavage"

# Main Streamlit app
def main():
    st.set_page_config(page_title="Analyse et Optimisation des Lavages de Puits", layout="wide")
    
    # Create an instance of the analysis class
    analyzer = WellWashingAnalysis()

    # App title
    st.title("Analyse et Optimisation des Lavages de Puits")
    
    # Sidebar for data loading
    st.sidebar.header("Chargement des données")
    
    data_option = st.sidebar.radio(
        "Source de données:",
        ["Utiliser les données d'exemple", "Charger un fichier Excel"]
    )
    
    # Load data based on selection
    if data_option == "Charger un fichier Excel":
        uploaded_file = st.sidebar.file_uploader("Choisir un fichier Excel", type=["xlsx", "xls"])
        if uploaded_file is not None:
            try:
                # Try different engines for Excel files
                try:
                    # First try with openpyxl (for .xlsx)
                    analyzer.df = pd.read_excel(uploaded_file, engine='openpyxl')
                except Exception:
                    # If that fails, try with xlrd (for .xls)
                    # Reset file position to beginning
                    uploaded_file.seek(0)
                    analyzer.df = pd.read_excel(uploaded_file, engine='xlrd')
                
                # Rename columns to make them easier to handle
                analyzer.df.columns = [col.replace(' ', '_') for col in analyzer.df.columns]
                st.sidebar.success("Fichier chargé avec succès!")
            except Exception as e:
                st.sidebar.error(f"Erreur lors du chargement du fichier: {str(e)}")
                # Add more detailed error information
                st.sidebar.info("Conseil: Vérifiez que votre fichier est au format Excel valide (.xlsx ou .xls). Si vous utilisez un format particulier, essayez de l'exporter en Excel standard.")
        else:
            st.sidebar.info("Veuillez télécharger un fichier Excel ou utiliser les données d'exemple.")
            analyzer.df = None
    else:
        analyzer.df = analyzer.df_sample.copy()
        st.sidebar.success("Données d'exemple chargées!")

    # Main content tabs
    if analyzer.df is not None:
        tab1, tab2, tab3 = st.tabs(["Données", "Analyse statistique", "Optimisation des lavages"])
        
        with tab1:
            st.header("Visualisation des données")
            st.dataframe(analyzer.df, use_container_width=True)
            
            # Download button for data
            csv = analyzer.df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger les données en CSV",
                data=csv,
                file_name="donnees_lavage_puits.csv",
                mime="text/csv",
            )
        
        with tab2:
            st.header("Analyse statistique")
            
            graph_type = st.selectbox(
                "Type de graphique:",
                [
                    "Fréquence des lavages par puits", 
                    "Heures d'arrêt moyennes par puits", 
                    "MAP moyen par puits",
                    "Corrélation heures d'arrêt / MAP",
                    "Distribution des heures d'arrêt"
                ]
            )
            
            # Generate the selected graph
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if graph_type == "Fréquence des lavages par puits":
                # Count the number of washings per well
                washings_per_well = analyzer.df['Puits'].value_counts().reset_index()
                washings_per_well.columns = ['Puits', 'Nombre de lavages']
                
                # Plot the graph
                sns.barplot(x='Puits', y='Nombre de lavages', data=washings_per_well, ax=ax)
                ax.set_title('Fréquence des lavages par puits')
                ax.set_xlabel('Puits')
                ax.set_ylabel('Nombre de lavages')
                ax.tick_params(axis='x', rotation=90)
                
            elif graph_type == "Heures d'arrêt moyennes par puits":
                # Calculate average downtime hours per well
                mean_hours = analyzer.df.groupby('Puits')['Heures_d_Arret'].mean().reset_index()
                
                # Plot the graph
                sns.barplot(x='Puits', y='Heures_d_Arret', data=mean_hours, ax=ax)
                ax.set_title("Heures d'arrêt moyennes par puits")
                ax.set_xlabel('Puits')
                ax.set_ylabel("Heures d'arrêt moyennes")
                ax.tick_params(axis='x', rotation=90)
                
            elif graph_type == "MAP moyen par puits":
                # Calculate average MAP per well
                mean_map = analyzer.df.groupby('Puits')['MAP'].mean().reset_index()
                
                # Plot the graph
                sns.barplot(x='Puits', y='MAP', data=mean_map, ax=ax)
                ax.set_title("MAP moyen par puits")
                ax.set_xlabel('Puits')
                ax.set_ylabel("MAP moyen (m³)")
                ax.tick_params(axis='x', rotation=90)
                
            elif graph_type == "Corrélation heures d'arrêt / MAP":
                # Plot the scatter plot
                sns.scatterplot(x='Heures_d_Arret', y='MAP', hue='Puits', data=analyzer.df, ax=ax)
                ax.set_title("Corrélation entre heures d'arrêt et MAP")
                ax.set_xlabel("Heures d'arrêt")
                ax.set_ylabel("MAP (m³)")
                
                # Add a trend line
                sns.regplot(x='Heures_d_Arret', y='MAP', data=analyzer.df, scatter=False, ax=ax, color='red')
                
            elif graph_type == "Distribution des heures d'arrêt":
                # Plot the histogram
                sns.histplot(data=analyzer.df, x='Heures_d_Arret', bins=10, kde=True, ax=ax)
                ax.set_title("Distribution des heures d'arrêt")
                ax.set_xlabel("Heures d'arrêt")
                ax.set_ylabel("Fréquence")
            
            # Adjust layout
            plt.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
            
            # Save graph button
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            st.download_button(
                label="Télécharger le graphique",
                data=buffer,
                file_name=f"{graph_type.replace(' ', '_').lower()}.png",
                mime="image/png"
            )
        
        with tab3:
            st.header("Optimisation des lavages")
            
            optim_method = st.selectbox(
                "Méthode d'optimisation:",
                [
                    "Réduction des heures d'arrêt",
                    "Maximisation du MAP par heure d'arrêt",
                    "Planification optimale des lavages"
                ]
            )
            
            if optim_method == "Réduction des heures d'arrêt":
                st.subheader("Optimisation des heures d'arrêt")
                
                # Run optimization
                results = analyzer.optimize_downtime_hours()
                
                # Display results
                st.write("Cette analyse identifie les puits qui pourraient bénéficier d'une réduction des heures d'arrêt.")
                
                # Priority wells
                st.subheader("Puits prioritaires pour l'optimisation:")
                
                for i, row in results.head(5).iterrows():
                    if row['Potentiel de réduction'] > 0:
                        with st.expander(f"Puits {row['Puits']}"):
                            st.write(f"- Durée moyenne des arrêts: {row['Moyenne']:.2f} heures")
                            st.write(f"- Durée minimale observée: {row['Minimum']} heures")
                            st.write(f"- Potentiel de réduction par lavage: {row['Potentiel de réduction']:.2f} heures")
                            st.write(f"- Économie potentielle totale: {row['Économie potentielle (heures)']:.2f} heures")
                            
                            # Analyze influence factors
                            well_data = analyzer.df[analyzer.df['Puits'] == row['Puits']]
                            
                            # Check if there's a correlation between MAP and downtime hours
                            if len(well_data) > 2:  # Enough data for a correlation
                                corr = well_data['MAP'].corr(well_data['Heures_d_Arret'])
                                
                                if abs(corr) > 0.5:
                                    st.write(f"- Forte corrélation ({corr:.2f}) entre MAP et heures d'arrêt pour ce puits")
                                    if corr > 0:
                                        st.write("  → Les lavages avec MAP élevé prennent plus de temps")
                                    else:
                                        st.write("  → Les lavages avec MAP élevé prennent moins de temps (à vérifier)")
                
                # General recommendations
                total_hours = results['Total'].sum()
                total_savings = results['Économie potentielle (heures)'].sum()
                
                # Display summary in a metrics section
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Heures d'arrêt totales", f"{total_hours:.1f} h")
                with col2:
                    st.metric("Économie potentielle", f"{total_savings:.1f} h")
                with col3:
                    st.metric("Économie en pourcentage", f"{(total_savings/total_hours*100):.1f}%")
                
                st.subheader("Recommandations:")
                st.markdown("""
                1. Standardiser les procédures de lavage pour les puits prioritaires
                2. Étudier les pratiques optimales des puits avec des temps d'arrêt minimaux
                3. Mettre en place un suivi des temps d'arrêt et des KPIs de performance
                """)
                
                # Download results button
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger les résultats en CSV",
                    data=csv,
                    file_name="optimisation_heures_arret.csv",
                    mime="text/csv",
                )
            
            elif optim_method == "Maximisation du MAP par heure d'arrêt":
                st.subheader("Optimisation du MAP par heure d'arrêt")
                
                # Run optimization
                results = analyzer.optimize_map_per_hour()
                
                # Display results
                st.write("Cette analyse identifie les puits où l'efficacité des lavages peut être améliorée.")
                
                # Wells with the best improvement potential
                st.subheader("Puits avec le meilleur potentiel d'amélioration:")
                
                for i, row in results.head(5).iterrows():
                    if row['Potentiel d\'amélioration'] > 0:
                        with st.expander(f"Puits {row['Puits']}"):
                            st.write(f"- Efficacité moyenne: {row['Efficacité moyenne']:.2f} m³/heure")
                            st.write(f"- Meilleure efficacité observée: {row['Efficacité max']:.2f} m³/heure")
                            potentiel = row['Potentiel d\'amélioration']
                            potentiel_pct = row['Potentiel en %']
                            st.write(f"- Potentiel d'amélioration: {potentiel:.2f} m³/heure ({potentiel_pct:.2f}%)")
                            
                            # Examine the best washing for this well
                            well_data = analyzer.df[analyzer.df['Puits'] == row['Puits']]
                            best_washing = well_data.loc[well_data['Efficacité'].idxmax()]
                            
                            st.write("- Le lavage le plus efficace pour ce puits:")
                            st.write(f"  Date: {best_washing['Date_de_Fermeture'] if pd.notna(best_washing['Date_de_Fermeture']) else 'Non spécifiée'}")
                            st.write(f"  MAP: {best_washing['MAP']:.2f} m³")
                            st.write(f"  Heures d'arrêt: {best_washing['Heures_d_Arret']} heures")
                            st.write(f"  Efficacité: {best_washing['Efficacité']:.2f} m³/heure")
                
                # Most efficient well
                best_well = results.loc[results['Efficacité moyenne'].idxmax()]
                
                st.subheader("Puits le plus efficace:")
                st.write(f"Puits {best_well['Puits']}:")
                st.write(f"- Efficacité moyenne: {best_well['Efficacité moyenne']:.2f} m³/heure")
                st.write(f"- MAP moyen: {best_well['MAP moyen']:.2f} m³")
                
                st.subheader("Recommandations:")
                st.markdown("""
                1. Étudier les pratiques utilisées lors des lavages les plus efficaces
                2. Standardiser les méthodes de lavage basées sur les meilleures performances
                3. Mettre en place un suivi de l'efficacité (MAP/heure) comme KPI principal
                4. Former les équipes aux meilleures pratiques identifiées
                """)
                
                # Download results button
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger les résultats en CSV",
                    data=csv,
                    file_name="optimisation_map_par_heure.csv",
                    mime="text/csv",
                )
            
            elif optim_method == "Planification optimale des lavages":
                st.subheader("Planification optimale des lavages")
                
                # Run optimization
                results, error = analyzer.plan_washings()
                
                if error:
                    st.error(error)
                elif results:
                    # Display observed frequency
                    st.markdown("### Fréquence de lavage observée:")
                    
                    for well_info in results['frequency_data']:
                        with st.expander(f"Puits {well_info['well']}"):
                            st.write(f"- Intervalle moyen entre les lavages: {well_info['mean_interval']:.1f} jours")
                            
                            if well_info['std_interval'] is not None:
                                st.write(f"- Écart-type: {well_info['std_interval']:.1f} jours")
                            
                            st.write(f"- Dates observées: {', '.join(well_info['dates'])}")
                    
                    # Planned washings
                    st.markdown("### Planification des prochains lavages:")
                    
                    # Convert datetime to string for display
                    display_df = results['planned_dates'].copy()
                    display_df['Dernier_lavage'] = display_df['Dernier_lavage'].dt.strftime('%d/%m/%Y')
                    display_df['Prochain_lavage'] = display_df['Prochain_lavage'].dt.strftime('%d/%m/%Y')
                    
                    st.dataframe(display_df[['Puits', 'Dernier_lavage', 'Prochain_lavage', 'Jours_avant', 'Priorité']], use_container_width=True)
                    
                    # Group washings by week for better planning
                    st.markdown("### Regroupement par semaine:")
                    
                    # Convert string dates back to datetime for processing
                    display_df['Prochain_lavage'] = pd.to_datetime(display_df['Prochain_lavage'], format='%d/%m/%Y')
                    
                    # Add week number to dataframe
                    display_df['Semaine'] = display_df['Prochain_lavage'].dt.isocalendar().week
                    display_df['Année'] = display_df['Prochain_lavage'].dt.isocalendar().year
                    
                    # Group by week and count
                    weekly_count = display_df.groupby(['Année', 'Semaine']).size().reset_index(name='Nombre de lavages')
                    
                    # Convert back to string for display
                    display_df['Prochain_lavage'] = display_df['Prochain_lavage'].dt.strftime('%d/%m/%Y')
                    
                    for _, row in weekly_count.iterrows():
                        week_wells = display_df[(display_df['Semaine'] == row['Semaine']) & 
                                              (display_df['Année'] == row['Année'])]
                        
                        st.write(f"**Semaine {row['Semaine']} de {row['Année']}:** {row['Nombre de lavages']} lavages planifiés")
                        st.dataframe(week_wells[['Puits', 'Prochain_lavage', 'Priorité']], use_container_width=True)
                    
                    # Highlight wells that need to be washed in the next 7 days
                    urgent_wells = display_df[display_df['Jours_avant'] <= 7]
                    
                    if not urgent_wells.empty:
                        st.markdown("### Lavages urgents (7 jours):")
                        st.dataframe(urgent_wells[['Puits', 'Prochain_lavage', 'Jours_avant', 'Priorité']], use_container_width=True)
                    
                    # Check if there are wells with close washing dates
                    if results['close_washings']:
                        st.markdown("### Lavages rapprochés (opportunités de regroupement):")
                        for well1, well2 in results['close_washings']:
                            st.write(f"Les puits {well1} et {well2} peuvent être lavés ensemble (dates rapprochées)")
                    
                    # Download results button
                    csv = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Télécharger le planning en CSV",
                        data=csv,
                        file_name="planning_lavages.csv",
                        mime="text/csv",
                    )
                    
                    # Add a calendar view
                    st.markdown("### Calendrier des lavages:")
                    
                    # Create a calendar view with HTML and CSS
                    cal_html = """
                    <style>
                    .calendar {
                        width: 100%;
                        border-collapse: collapse;
                    }
                    .calendar th {
                        background-color: #f2f2f2;
                        padding: 10px;
                        text-align: center;
                        border: 1px solid #ddd;
                    }
                    .calendar td {
                        height: 80px;
                        vertical-align: top;
                        width: 14.28%;
                        border: 1px solid #ddd;
                        padding: 5px;
                    }
                    .calendar td.other-month {
                        background-color: #f9f9f9;
                        color: #999;
                    }
                    .calendar .event {
                        background-color: #4CAF50;
                        color: white;
                        padding: 3px;
                        margin-bottom: 3px;
                        border-radius: 3px;
                        font-size: 12px;
                    }
                    .calendar .event.high {
                        background-color: #f44336;
                    }
                    .calendar .event.medium {
                        background-color: #ff9800;
                    }
                    </style>
                    """
                    
                    # Add a simple visualization using Streamlit components
                    today = datetime.now()
                    start_date = today - pd.Timedelta(days=today.weekday())
                    end_date = start_date + pd.Timedelta(days=41)  # 6 weeks
                    
                    calendar_events = []
                    
                    # Convert to datetime for processing if necessary
                    if isinstance(display_df['Prochain_lavage'].iloc[0], str):
                        display_df['Prochain_lavage'] = pd.to_datetime(display_df['Prochain_lavage'], format='%d/%m/%Y')
                    
                    # Create events for the calendar
                    for _, row in display_df.iterrows():
                        date = row['Prochain_lavage']
                        if start_date <= date <= end_date:
                            priority_class = ""
                            if "HAUTE" in row['Priorité']:
                                priority_class = "high"
                            elif "MOYENNE" in row['Priorité']:
                                priority_class = "medium"
                            
                            calendar_events.append({
                                'date': date,
                                'well': row['Puits'],
                                'priority': priority_class
                            })
                    
                    # Generate 6-week calendar
                    weeks = []
                    current_date = start_date
                    
                    while current_date < end_date:
                        week = []
                        for _ in range(7):
                            day_events = [event for event in calendar_events if event['date'].date() == current_date.date()]
                            week.append({
                                'date': current_date,
                                'events': day_events,
                                'is_other_month': current_date.month != today.month
                            })
                            current_date += pd.Timedelta(days=1)
                        weeks.append(week)
                    
                    # Format the calendar HTML
                    cal_html += '<table class="calendar"><tr><th>Lun</th><th>Mar</th><th>Mer</th><th>Jeu</th><th>Ven</th><th>Sam</th><th>Dim</th></tr>'
                    
                    for week in weeks:
                        cal_html += '<tr>'
                        for day in week:
                            other_month_class = ' class="other-month"' if day['is_other_month'] else ''
                            cal_html += f'<td{other_month_class}><div>{day["date"].day}</div>'
                            
                            for event in day['events']:
                                event_class = f'event {event["priority"]}' if event['priority'] else 'event'
                                cal_html += f'<div class="{event_class}">{event["well"]}</div>'
                                
                            cal_html += '</td>'
                        cal_html += '</tr>'
                    
                    cal_html += '</table>'
                    
                    st.markdown(cal_html, unsafe_allow_html=True)
                    
                    # Recommendations
                    st.subheader("Recommandations:")
                    st.markdown("""
                    1. Planifier les lavages selon le calendrier proposé
                    2. Regrouper les lavages proches pour optimiser la logistique
                    3. Donner la priorité aux puits à haute priorité (en retard)
                    4. Mettre à jour régulièrement les données pour affiner la planification
                    5. Considérer l'historique d'efficacité des lavages précédents
                    """)

if __name__ == "__main__":
    main()
