
# Run the Q&A system (non-interactive demo portion)
exec(open('dream_qa.py').read().replace('interactive_qa(model, word_to_id, id_to_word)', 
                                         'print("\\n✓ Interactive mode ready (skipped in demo)")'))
