# Document Inventory — Streaming Live RAG Development/Demo Corpus

| Doc_ID | Filename | Purpose | # Sections | Key Facts | Relationships |
|---|---|---|---|---|---|
| Doc_01 | doc_01_venue_grandhall_pune.md | Pune venue: small/mid workshops | 8 | Capacity 40 seated/45 standing; INR 18k/30k half/full day | Cancellation terms cross-ref Doc_04 §5; catering cross-ref Doc_05/07 |
| Doc_02 | doc_02_venue_riverside_pune.md | Pune venue: large-format events | 7 | Capacity 80 seated/100 standing; INR 55k full day | Cancellation terms cross-ref Doc_04 §5; alt. to Doc_01 for late-detail capacity upgrade |
| Doc_03 | doc_03_venue_bangalore_techpark.md | Bangalore venue: large-scale events | 8 | Room B-204 (100 seated), combined 180 standing; hackathon Wi-Fi | Cancellation terms cross-ref Doc_04 §5; catering cross-ref Doc_06 |
| Doc_04 | doc_04_event_booking_policy_general.md | Uniform cancellation/booking policy across all venues | 6 | 14-day full refund / <14-day 50% forfeit; INR 50k approval threshold | Authoritative source for cancellation claims in Doc_01/02/03 |
| Doc_05 | doc_05_catering_spice_route.md | Pune-primary catering vendor | 7 | Packages A/B/C, INR 250/450/650 per head | Serves Doc_01, Doc_02; Bangalore surcharge INR 3,000 |
| Doc_06 | doc_06_catering_urban_bites.md | Bangalore-primary catering vendor | 7 | Standard/Premium, INR 350/700 per head | Serves Doc_03 |
| Doc_07 | doc_07_catering_policy_general.md | Uniform catering ordering/budget policy | 6 | Budget caps INR 400 (internal) / INR 800 (client-facing) per head | Governs Doc_05, Doc_06 vendor engagement |
| Doc_08 | doc_08_travel_expense_domestic.md | Domestic travel reimbursement rules | 8 | 30-day receipt window; per diem INR 1,200/900 | CONFLICTS with Doc_11 §2 (receipt window); base policy for Doc_09 |
| Doc_09 | doc_09_travel_expense_international.md | International travel reimbursement rules | 8 | 14-day late-booking director-approval exception; per diem USD 60 | Supplements Doc_08; late-detail pivot doc for domestic→international scenarios |
| Doc_10 | doc_10_expense_approval_workflow.md | Expense claim routing/turnaround | 6 | Approval chain by amount; 3+5 business day turnaround | Applies uniformly to Doc_08, Doc_09 claims |
| Doc_11 | doc_11_travel_policy_addendum_2026.md | 2026 memo updating receipt window | 5 | Reduces receipt window 30→15 days, effective next fiscal quarter | CONFLICTS with Doc_08 §3; supersedes by effective date |
| Doc_12 | doc_12_employee_handbook_onboarding.md | New-hire onboarding process | 7 | IT security training within 5 working days | NEAR-DUPLICATE of Doc_17 §3 (security training requirement) |
| Doc_13 | doc_13_employee_handbook_leave_policy.md | Leave entitlements | 8 | 12 casual / 10 sick / 18 earned days per year | Standalone |
| Doc_14 | doc_14_employee_handbook_resignation_v3.md | Older resignation notice policy | 5 | 30-day notice period, all employees | CONFLICTS with Doc_15 §2 (superseded by 2026 memo) |
| Doc_15 | doc_15_hr_policy_update_memo_2026.md | 2026 memo revising notice period | 4 | 45-day notice period, all employees, supersedes Doc_14 | CONFLICTS with Doc_14 §2 |
| Doc_16 | doc_16_it_security_password_access.md | Password/access policy (SEC-014) | 7 | 12-char passwords; 90-day (60-day admin) rotation; MFA required | BM25/lexical-friendly exact codes and numbers |
| Doc_17 | doc_17_it_security_device_onboarding.md | Device registration/security policy | 6 | New device registration; security training within 5 working days | NEAR-DUPLICATE of Doc_12 §3 |
| Doc_18 | doc_18_remote_work_policy.md | Remote/flexible work eligibility | 7 | Core hours 11am-4pm; INR 15k equipment stipend; INR 5k/mo co-working | Dense/semantic-friendly abstract phrasing |
| Doc_19 | doc_19_office_facilities_bangalore.md | Bangalore campus facilities | 7 | 150-vehicle parking; INR 150/day meal subsidy; NO gym mentioned | THIN COVERAGE: no fitness facility (contrast with Doc_20 §5) |
| Doc_20 | doc_20_office_facilities_pune.md | Pune campus facilities | 6 | 80-vehicle parking; on-site fitness room 6am-10pm | Fitness room present here only, not in Doc_19 |
| Doc_21 | doc_21_visa_documentation_policy.md | Business-travel visa support | 4 | Invitation letter support, 20 business days lead time | Adjacent to but distinct from Doc_09; NOT relocation assistance |
| Doc_22 | doc_22_meeting_room_booking_general.md | Standard meeting room booking (non-event-venue) | 5 | No approval needed <12 people; 15-min no-show release | Distinct from Doc_01/02/03/04 (event venues, not standard rooms) |
| Doc_23 | doc_23_vendor_code_of_conduct.md | Vendor conduct standards | 5 | Anti-bribery; confidentiality; 30-day termination notice | Applies to Doc_05, Doc_06 vendor relationships |
| Doc_24 | doc_24_data_privacy_policy.md | Data classification/retention | 6 | 7-year retention for Confidential data (expense/attendee records) | Governs data handling implied by Doc_08-11, Doc_23 |
