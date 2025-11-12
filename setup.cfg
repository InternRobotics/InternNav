[yapf]
based_on_style = pep8
blank_line_before_nested_class_or_def = true
split_before_expression_after_opening_paren = true

[isort]
line_length = 79
multi_line_output = 0
extra_standard_library = pkg_resources,setuptools
known_first_party = internutopia, internutopia_extension, grevaluator, grbench, grmodel
no_lines_before = STDLIB,LOCALFOLDER
default_section = THIRDPARTY
skip_glob = internutopia/*, internutopia_extension/*, internnav/scripts/eval/configs/*


# ignore-words-list needs to be lowercase format. For example, if we want to
# ignore word "BA", then we need to append "ba" to ignore-words-list rather
# than "BA"
[codespell]
quiet-level = 3
ignore-words-list = patten,nd,ty,mot,hist,formating,jetbot,wth,coverted,descrete,thw,ro
skip =
    *.js
    *.txt
    *.md
    *.json
    *.ipynb


[flake8]
show-source=True
statistics=True
per-file-ignores=*/__init__.py:F401
# E402: Module level import not at top of file
# E501: Line too long
# W503: Line break before binary operator
# E203: Whitespace before ':' -> conflicts with black
# D401: First line should be in imperative mood
# R504: Unnecessary variable assignment before return statement.
# R505: Unnecessary elif after return statement
# SIM102: Use a single if-statement instead of nested if-statements
# SIM117: Merge with statements for context managers that have same scope.
# E711: comparison to None should be 'if cond is None:'
# E226: missing whitespace around arithmetic operator
ignore=E402,E501,W503,E203,D401,R504,R505,SIM102,SIM117,E711,E226
max-line-length = 120
max-complexity = 30
exclude=_*,.vscode,.git,docs/**,**/test/**,**/lcmtypes/**,*.ipynb
