
GSE_data_categorical_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']

GSE_new_colums = ['x0_Exactly true', 'x0_Hardly true', 'x0_Moderately true', 'x0_Not at all true', 'x1_Exactly true',
              'x1_Hardly true', 'x1_Moderately true', 'x1_Not at all true', 'x2_Exactly true', 'x2_Hardly true',
              'x2_Moderately true', 'x2_Not at all true', 'x3_Exactly true', 'x3_Hardly true', 'x3_Moderately true',
              'x3_Not at all true', 'x4_Exactly true', 'x4_Hardly true', 'x4_Moderately true', 'x4_Not at all true',
              'x5_Exactly true', 'x5_Hardly true', 'x5_Moderately true', 'x5_Not at all true', 'x6_Exactly true',
              'x6_Hardly true', 'x6_Moderately true', 'x6_Not at all true', 'x7_Exactly true', 'x7_Hardly true',
              'x7_Moderately true', 'x7_Not at all true', 'x8_Exactly true', 'x8_Hardly true', 'x8_Moderately true',
              'x8_Not at all true', 'x9_Exactly true', 'x9_Hardly true', 'x9_Moderately true', 'x9_Not at all true']


Anxiety_data_categorical_columns = ['Q2A', 'Q4A', 'Q7A', 'Q9A', 'Q15A', 'Q19A', 'Q20A', 'Q23A', 'Q25A', 'Q28A', 'Q30A',
                                    'Q36A', 'Q40A', 'Q41A', 'Extraverted-enthusiastic', 'Critical-quarrelsome',
                                    'Dependable-self_disciplined', 'Anxious-easily upset',
                                    'Open to new experiences-complex', 'Reserved-quiet', 'Sympathetic-warm',
                                    'Disorganized-careless', 'Calm-emotionally_stable', 'Conventional-uncreative',
                                    'education', 'gender', 'engnat', 'screensize', 'hand', 'religion', 'orientation',
                                    'married', 'Age_Groups']

Anxiety_new_colums = ['x0_ALMOST ALWAYS', 'x0_NEVER', 'x0_OFTEN', 'x0_SOMETIMES', 'x1_ALMOST ALWAYS', 'x1_NEVER',
                      'x1_OFTEN',
                      'x1_SOMETIMES', 'x2_ALMOST ALWAYS', 'x2_NEVER', 'x2_OFTEN', 'x2_SOMETIMES', 'x3_ALMOST ALWAYS',
                      'x3_NEVER', 'x3_OFTEN', 'x3_SOMETIMES', 'x4_ALMOST ALWAYS', 'x4_NEVER', 'x4_OFTEN',
                      'x4_SOMETIMES',
                      'x5_ALMOST ALWAYS', 'x5_NEVER', 'x5_OFTEN', 'x5_SOMETIMES', 'x6_ALMOST ALWAYS', 'x6_NEVER',
                      'x6_OFTEN',
                      'x6_SOMETIMES', 'x7_ALMOST ALWAYS', 'x7_NEVER', 'x7_OFTEN', 'x7_SOMETIMES', 'x8_ALMOST ALWAYS',
                      'x8_NEVER', 'x8_OFTEN', 'x8_SOMETIMES', 'x9_ALMOST ALWAYS', 'x9_NEVER', 'x9_OFTEN',
                      'x9_SOMETIMES',
                      'x10_ALMOST ALWAYS', 'x10_NEVER', 'x10_OFTEN', 'x10_SOMETIMES', 'x11_ALMOST ALWAYS', 'x11_NEVER',
                      'x11_OFTEN', 'x11_SOMETIMES', 'x12_ALMOST ALWAYS', 'x12_NEVER', 'x12_OFTEN', 'x12_SOMETIMES',
                      'x13_ALMOST ALWAYS', 'x13_NEVER', 'x13_OFTEN', 'x13_SOMETIMES', 'x14_Agree a little',
                      'x14_Agree moderately', 'x14_Agree strongly', 'x14_Disagree a little', 'x14_Disagree moderately',
                      'x14_Disagree strongly', 'x14_Neither agree nor disagree', 'x15_Agree a little',
                      'x15_Agree moderately',
                      'x15_Agree strongly', 'x15_Disagree a little', 'x15_Disagree moderately', 'x15_Disagree strongly',
                      'x15_Neither agree nor disagree', 'x16_Agree a little', 'x16_Agree moderately',
                      'x16_Agree strongly',
                      'x16_Disagree a little', 'x16_Disagree moderately', 'x16_Disagree strongly',
                      'x16_Neither agree nor disagree', 'x17_Agree a little', 'x17_Agree moderately',
                      'x17_Agree strongly',
                      'x17_Disagree a little', 'x17_Disagree moderately', 'x17_Disagree strongly',
                      'x17_Neither agree nor disagree', 'x18_Agree a little', 'x18_Agree moderately',
                      'x18_Agree strongly',
                      'x18_Disagree a little', 'x18_Disagree moderately', 'x18_Disagree strongly',
                      'x18_Neither agree nor disagree', 'x19_Agree a little', 'x19_Agree moderately',
                      'x19_Agree strongly',
                      'x19_Disagree a little', 'x19_Disagree moderately', 'x19_Disagree strongly',
                      'x19_Neither agree nor disagree', 'x20_Agree a little', 'x20_Agree moderately',
                      'x20_Agree strongly',
                      'x20_Disagree a little', 'x20_Disagree moderately', 'x20_Disagree strongly',
                      'x20_Neither agree nor disagree', 'x21_Agree a little', 'x21_Agree moderately',
                      'x21_Agree strongly',
                      'x21_Disagree a little', 'x21_Disagree moderately', 'x21_Disagree strongly',
                      'x21_Neither agree nor disagree', 'x22_Agree a little', 'x22_Agree moderately',
                      'x22_Agree strongly',
                      'x22_Disagree a little', 'x22_Disagree moderately', 'x22_Disagree strongly',
                      'x22_Neither agree nor disagree', 'x23_Agree a little', 'x23_Agree moderately',
                      'x23_Agree strongly',
                      'x23_Disagree a little', 'x23_Disagree moderately', 'x23_Disagree strongly',
                      'x23_Neither agree nor disagree', 'x24_Graduate degree', 'x24_High school',
                      'x24_Less than high school',
                      'x24_University degree', 'x25_Female', 'x25_Male', 'x25_Other', 'x26_3', 'x26_No', 'x26_Yes',
                      'x27_laptop or desktop', 'x27_phone', 'x28_Both', 'x28_Left', 'x28_Right', 'x29_Agnostic',
                      'x29_Atheist',
                      'x29_Buddhist', 'x29_Christian (Catholic)', 'x29_Christian (Mormon)', 'x29_Christian (Other)',
                      'x29_Christian (Protestant)', 'x29_Hindu', 'x29_Jewish', 'x29_Muslim', 'x29_Other', 'x29_Sikh',
                      'x30_Asexual', 'x30_Bisexual', 'x30_Heterosexual', 'x30_Homosexual', 'x30_Other',
                      'x31_Currently married',
                      'x31_Never married', 'x31_Previously married', 'x32_ Primary Children', 'x32_Adults',
                      'x32_Elder Adults',
                      'x32_Older People', 'x32_Secondary Children']

Depression_data_categorical_columns = ['Q3A', 'Q5A', 'Q10A', 'Q13A', 'Q16A', 'Q17A', 'Q21A', 'Q24A', 'Q26A',
                                       'Q31A', 'Q34A', 'Q37A', 'Q38A', 'Q42A', 'Extraverted-enthusiastic',
                                       'Critical-quarrelsome', 'Dependable-self_disciplined',
                                       'Anxious-easily upset', 'Open to new experiences-complex',
                                       'Reserved-quiet', 'Sympathetic-warm', 'Disorganized-careless',
                                       'Calm-emotionally_stable', 'Conventional-uncreative', 'education',
                                       'gender', 'engnat', 'screensize', 'hand', 'religion', 'orientation',
                                       'married', 'Age_Groups']

Depression_new_colums = ['x0_ALMOST ALWAYS', 'x0_NEVER', 'x0_OFTEN', 'x0_SOMETIMES', 'x1_ALMOST ALWAYS', 'x1_NEVER',
                         'x1_OFTEN',
                         'x1_SOMETIMES', 'x2_ALMOST ALWAYS', 'x2_NEVER', 'x2_OFTEN', 'x2_SOMETIMES', 'x3_ALMOST ALWAYS',
                         'x3_NEVER', 'x3_OFTEN', 'x3_SOMETIMES', 'x4_ALMOST ALWAYS', 'x4_NEVER', 'x4_OFTEN',
                         'x4_SOMETIMES',
                         'x5_ALMOST ALWAYS', 'x5_NEVER', 'x5_OFTEN', 'x5_SOMETIMES', 'x6_ALMOST ALWAYS', 'x6_NEVER',
                         'x6_OFTEN',
                         'x6_SOMETIMES', 'x7_ALMOST ALWAYS', 'x7_NEVER', 'x7_OFTEN', 'x7_SOMETIMES', 'x8_ALMOST ALWAYS',
                         'x8_NEVER', 'x8_OFTEN', 'x8_SOMETIMES', 'x9_ALMOST ALWAYS', 'x9_NEVER', 'x9_OFTEN',
                         'x9_SOMETIMES',
                         'x10_ALMOST ALWAYS', 'x10_NEVER', 'x10_OFTEN', 'x10_SOMETIMES', 'x11_ALMOST ALWAYS',
                         'x11_NEVER',
                         'x11_OFTEN', 'x11_SOMETIMES', 'x12_ALMOST ALWAYS', 'x12_NEVER', 'x12_OFTEN', 'x12_SOMETIMES',
                         'x13_ALMOST ALWAYS', 'x13_NEVER', 'x13_OFTEN', 'x13_SOMETIMES', 'x14_Agree a little',
                         'x14_Agree moderately', 'x14_Agree strongly', 'x14_Disagree a little',
                         'x14_Disagree moderately',
                         'x14_Disagree strongly', 'x14_Neither agree nor disagree', 'x15_Agree a little',
                         'x15_Agree moderately',
                         'x15_Agree strongly', 'x15_Disagree a little', 'x15_Disagree moderately',
                         'x15_Disagree strongly',
                         'x15_Neither agree nor disagree', 'x16_Agree a little', 'x16_Agree moderately',
                         'x16_Agree strongly',
                         'x16_Disagree a little', 'x16_Disagree moderately', 'x16_Disagree strongly',
                         'x16_Neither agree nor disagree', 'x17_Agree a little', 'x17_Agree moderately',
                         'x17_Agree strongly',
                         'x17_Disagree a little', 'x17_Disagree moderately', 'x17_Disagree strongly',
                         'x17_Neither agree nor disagree', 'x18_Agree a little', 'x18_Agree moderately',
                         'x18_Agree strongly',
                         'x18_Disagree a little', 'x18_Disagree moderately', 'x18_Disagree strongly',
                         'x18_Neither agree nor disagree', 'x19_Agree a little', 'x19_Agree moderately',
                         'x19_Agree strongly',
                         'x19_Disagree a little', 'x19_Disagree moderately', 'x19_Disagree strongly',
                         'x19_Neither agree nor disagree', 'x20_Agree a little', 'x20_Agree moderately',
                         'x20_Agree strongly',
                         'x20_Disagree a little', 'x20_Disagree moderately', 'x20_Disagree strongly',
                         'x20_Neither agree nor disagree', 'x21_Agree a little', 'x21_Agree moderately',
                         'x21_Agree strongly',
                         'x21_Disagree a little', 'x21_Disagree moderately', 'x21_Disagree strongly',
                         'x21_Neither agree nor disagree', 'x22_Agree a little', 'x22_Agree moderately',
                         'x22_Agree strongly',
                         'x22_Disagree a little', 'x22_Disagree moderately', 'x22_Disagree strongly',
                         'x22_Neither agree nor disagree', 'x23_Agree a little', 'x23_Agree moderately',
                         'x23_Agree strongly',
                         'x23_Disagree a little', 'x23_Disagree moderately', 'x23_Disagree strongly',
                         'x23_Neither agree nor disagree', 'x24_Graduate degree', 'x24_High school',
                         'x24_Less than high school',
                         'x24_University degree', 'x25_Female', 'x25_Male', 'x25_Other', 'x26_3', 'x26_No', 'x26_Yes',
                         'x27_laptop or desktop', 'x27_phone', 'x28_Both', 'x28_Left', 'x28_Right', 'x29_Agnostic',
                         'x29_Atheist',
                         'x29_Buddhist', 'x29_Christian (Catholic)', 'x29_Christian (Mormon)', 'x29_Christian (Other)',
                         'x29_Christian (Protestant)', 'x29_Hindu', 'x29_Jewish', 'x29_Muslim', 'x29_Other', 'x29_Sikh',
                         'x30_Asexual', 'x30_Bisexual', 'x30_Heterosexual', 'x30_Homosexual', 'x30_Other',
                         'x31_Currently married',
                         'x31_Never married', 'x31_Previously married', 'x32_Adults', 'x32_Elder Adults',
                         'x32_Older People',
                         'x32_Primary Children', 'x32_Secondary Children']

Stress_data_categorical_columns = ['Q1A', 'Q6A', 'Q8A', 'Q11A', 'Q12A', 'Q14A', 'Q18A', 'Q22A', 'Q27A',
                                   'Q29A', 'Q32A', 'Q33A', 'Q35A', 'Q39A', 'Extraverted-enthusiastic',
                                   'Critical-quarrelsome', 'Dependable-self_disciplined',
                                   'Anxious-easily upset', 'Open to new experiences-complex',
                                   'Reserved-quiet', 'Sympathetic-warm', 'Disorganized-careless',
                                   'Calm-emotionally_stable', 'Conventional-uncreative', 'education',
                                   'gender', 'engnat', 'screensize', 'hand', 'religion', 'orientation',
                                   'married', 'Age_Groups']

Stress_new_colums = ['x0_ALMOST ALWAYS', 'x0_NEVER', 'x0_OFTEN', 'x0_SOMETIMES', 'x1_ALMOST ALWAYS', 'x1_NEVER',
                     'x1_OFTEN',
                     'x1_SOMETIMES', 'x2_ALMOST ALWAYS', 'x2_NEVER', 'x2_OFTEN', 'x2_SOMETIMES', 'x3_ALMOST ALWAYS',
                     'x3_NEVER', 'x3_OFTEN', 'x3_SOMETIMES', 'x4_ALMOST ALWAYS', 'x4_NEVER', 'x4_OFTEN', 'x4_SOMETIMES',
                     'x5_ALMOST ALWAYS', 'x5_NEVER', 'x5_OFTEN', 'x5_SOMETIMES', 'x6_ALMOST ALWAYS', 'x6_NEVER',
                     'x6_OFTEN',
                     'x6_SOMETIMES', 'x7_ALMOST ALWAYS', 'x7_NEVER', 'x7_OFTEN', 'x7_SOMETIMES', 'x8_ALMOST ALWAYS',
                     'x8_NEVER', 'x8_OFTEN', 'x8_SOMETIMES', 'x9_ALMOST ALWAYS', 'x9_NEVER', 'x9_OFTEN', 'x9_SOMETIMES',
                     'x10_ALMOST ALWAYS', 'x10_NEVER', 'x10_OFTEN', 'x10_SOMETIMES', 'x11_ALMOST ALWAYS', 'x11_NEVER',
                     'x11_OFTEN', 'x11_SOMETIMES', 'x12_ALMOST ALWAYS', 'x12_NEVER', 'x12_OFTEN', 'x12_SOMETIMES',
                     'x13_ALMOST ALWAYS', 'x13_NEVER', 'x13_OFTEN', 'x13_SOMETIMES', 'x14_Agree a little',
                     'x14_Agree moderately', 'x14_Agree strongly', 'x14_Disagree a little', 'x14_Disagree moderately',
                     'x14_Disagree strongly', 'x14_Neither agree nor disagree', 'x15_Agree a little',
                     'x15_Agree moderately',
                     'x15_Agree strongly', 'x15_Disagree a little', 'x15_Disagree moderately', 'x15_Disagree strongly',
                     'x15_Neither agree nor disagree', 'x16_Agree a little', 'x16_Agree moderately',
                     'x16_Agree strongly',
                     'x16_Disagree a little', 'x16_Disagree moderately', 'x16_Disagree strongly',
                     'x16_Neither agree nor disagree', 'x17_Agree a little', 'x17_Agree moderately',
                     'x17_Agree strongly',
                     'x17_Disagree a little', 'x17_Disagree moderately', 'x17_Disagree strongly',
                     'x17_Neither agree nor disagree', 'x18_Agree a little', 'x18_Agree moderately',
                     'x18_Agree strongly',
                     'x18_Disagree a little', 'x18_Disagree moderately', 'x18_Disagree strongly',
                     'x18_Neither agree nor disagree', 'x19_Agree a little', 'x19_Agree moderately',
                     'x19_Agree strongly',
                     'x19_Disagree a little', 'x19_Disagree moderately', 'x19_Disagree strongly',
                     'x19_Neither agree nor disagree', 'x20_Agree a little', 'x20_Agree moderately',
                     'x20_Agree strongly',
                     'x20_Disagree a little', 'x20_Disagree moderately', 'x20_Disagree strongly',
                     'x20_Neither agree nor disagree', 'x21_Agree a little', 'x21_Agree moderately',
                     'x21_Agree strongly',
                     'x21_Disagree a little', 'x21_Disagree moderately', 'x21_Disagree strongly',
                     'x21_Neither agree nor disagree', 'x22_Agree a little', 'x22_Agree moderately',
                     'x22_Agree strongly',
                     'x22_Disagree a little', 'x22_Disagree moderately', 'x22_Disagree strongly',
                     'x22_Neither agree nor disagree', 'x23_Agree a little', 'x23_Agree moderately',
                     'x23_Agree strongly',
                     'x23_Disagree a little', 'x23_Disagree moderately', 'x23_Disagree strongly',
                     'x23_Neither agree nor disagree', 'x24_Graduate degree', 'x24_High school',
                     'x24_Less than high school',
                     'x24_University degree', 'x25_Female', 'x25_Male', 'x25_Other', 'x26_3', 'x26_No', 'x26_Yes',
                     'x27_laptop or desktop', 'x27_phone', 'x28_Both', 'x28_Left', 'x28_Right', 'x29_Agnostic',
                     'x29_Atheist',
                     'x29_Buddhist', 'x29_Christian (Catholic)', 'x29_Christian (Mormon)', 'x29_Christian (Other)',
                     'x29_Christian (Protestant)', 'x29_Hindu', 'x29_Jewish', 'x29_Muslim', 'x29_Other', 'x29_Sikh',
                     'x30_Asexual', 'x30_Bisexual', 'x30_Heterosexual', 'x30_Homosexual', 'x30_Other',
                     'x31_Currently married',
                     'x31_Never married', 'x31_Previously married', 'x32_ Primary Children', 'x32_Adults',
                     'x32_Elder Adults',
                     'x32_Older People', 'x32_Secondary Children']
