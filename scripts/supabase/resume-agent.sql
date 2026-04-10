create schema if not exists private;

create table if not exists private.resume_agent_profiles (
  id text primary key,
  profile jsonb not null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

alter table private.resume_agent_profiles enable row level security;

revoke all on schema private from public;
revoke all on all tables in schema private from public, anon, authenticated;

create or replace function public.create_resume_agent_link(p_profile jsonb)
returns text
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_id text;
begin
  loop
    v_id := substr(
      md5(random()::text || clock_timestamp()::text || coalesce(p_profile::text, '')),
      1,
      24
    );

    exit when not exists (
      select 1
      from private.resume_agent_profiles
      where id = v_id
    );
  end loop;

  insert into private.resume_agent_profiles (id, profile)
  values (v_id, p_profile);

  return v_id;
end;
$$;

create or replace function public.get_resume_agent_profile(p_id text)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_profile jsonb;
begin
  select profile
  into v_profile
  from private.resume_agent_profiles
  where id = lower(coalesce(p_id, ''))
  limit 1;

  return v_profile;
end;
$$;

grant execute on function public.create_resume_agent_link(jsonb) to anon, authenticated;
grant execute on function public.get_resume_agent_profile(text) to anon, authenticated;
