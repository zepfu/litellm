-- COHERE-002 originally created public.cohere_accepted_calls.
-- That unpublished table is superseded by the generic ledger
-- public.locally_counted_accepted_calls. COHERE-002 now uses the
-- generic ledger; apply the dated script below instead of this file.
--
-- See: scripts/apply_locally_counted_accepted_calls_2026_08_19.sql

\ir apply_locally_counted_accepted_calls_2026_08_19.sql
