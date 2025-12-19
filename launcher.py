
import sys
import os
from streamlit.web import cli as stcli

if __name__ == '__main__':
    # Construct the path to app.py
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    # Simulate arguments: "streamlit run app.py"
    # We use arguments compatible with direct execution
    sys.argv = ["streamlit", "run", app_path]
    
    print(f"Launching Streamlit App from: {sys.executable}")
    sys.exit(stcli.main())
