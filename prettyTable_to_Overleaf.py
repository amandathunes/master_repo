from prettytable import PrettyTable

# Make PrettyTable into Overleaf code

def overleaf_table_code(my_table: PrettyTable) -> str:
    # Initialize the LaTeX code string
    latex_code = "\\begin{tabular}{"
    
    # Determine the number of columns and construct the column specification
    num_columns = len(my_table.field_names)
    for i in range(num_columns):
        if i == 0:
            latex_code += "c|"
        elif i < num_columns-1:
            latex_code += "c|"
        elif i == num_columns-1:
            latex_code += "c"
        
    
    # Close the column specification and start the table body
    # latex_code += "}\n\\hline\n"
    latex_code += "}\n"
    
    header_row_entry = ""
    for field_name in my_table.field_names:
        header_row_entry += f"{field_name} & "
    header_row_entry = header_row_entry[:-2] + "\\\\ \\hline\n"
    
    latex_code += header_row_entry
    
    
    
    # Iterate through each row in the table
    i = len(my_table.rows)
    for row in my_table.rows:
        i -= 1
        # Construct the row entry, including column separators and line breaks
        row_entry = ""
        for idx, cell in enumerate(row):
            if isinstance(cell, str): 
                if "%" in cell:
                    cell = cell.replace("%", "\\%")
                elif "&" in cell:
                    cell = cell.replace("&", "\\&")
                elif "_" in cell:
                    cell = cell.replace("_", "\\_")
                elif "#" in cell:
                    cell = cell.replace("#", "\\#")
            if idx < len(row) - 1:
                row_entry += f"{cell} &  "
            else:
                row_entry += f"{cell}"
        if i == 0:
            row_entry += "\\\\ \n"
        else:
            row_entry += "\\\\ \\hline\n"
        
        # Append the constructed row entry to the LaTeX code
        latex_code += row_entry
    
    # End the table
    latex_code += "\\end{tabular}"
    
    return latex_code