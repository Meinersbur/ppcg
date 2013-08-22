#ifndef PPCG_H
#define PPCG_H

#include <isl/set.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <pet.h>

#include "ppcg_options.h"

int ppcg_extract_base_name(char *name, const char *input);

/* Representation of the scop for use inside PPCG.
 *
 * "options" are the options specified by the user.
 * Some fields in this structure may depend on some of the options.
 *
 * "start" and "end" are file offsets of the corresponding program text.
 * "context" represents constraints on the parameters.
 * "domain" is the union of all iteration domains.
 * "call" contains the iteration domains of statements with a call expression.
 * "reads" contains all potential read accesses.
 * "tagged_reads" is the same as "reads", except that the domain is a wrapped
 *	relation mapping an iteration domain to a reference identifier
 * "live_in" contains the potential read accesses that potentially
 *	have no corresponding writes in the scop.
 * "may_writes" contains all potential write accesses.
 * "tagged_may_writes" is the same as "may_writes", except that the domain
 *	is a wrapped relation mapping an iteration domain
 *	to a reference identifier
 * "must_writes" contains all definite write accesses.
 * "tagged_must_writes" is the same as "must_writes", except that the domain
 *	is a wrapped relation mapping an iteration domain
 *	to a reference identifier
 * "live_out" contains the potential write accesses that are potentially
 *	not killed by any kills or any other writes.
 * "tagged_must_kills" contains all definite kill accesses with
 *	a reference identifier in the domain.
 *
 * "tagger" maps iteration domains to the corresponding tagged
 *	iteration domain.
 *
 * "dep_flow" represents the potential flow dependences.
 * "dep_false" represents the potential false (anti and output) dependences.
 * "schedule" represents the (original) schedule.
 *
 * "types", "arrays" and "stmts" are copies of the corresponding elements
 * of the original pet_scop.
 */
struct ppcg_scop {
	struct ppcg_options *options;

	unsigned start;
	unsigned end;

	isl_set *context;
	isl_union_set *domain;
	isl_union_set *call;
	isl_union_map *tagged_reads;
	isl_union_map *reads;
	isl_union_map *live_in;
	isl_union_map *tagged_may_writes;
	isl_union_map *may_writes;
	isl_union_map *tagged_must_writes;
	isl_union_map *must_writes;
	isl_union_map *live_out;
	isl_union_map *tagged_must_kills;

	isl_union_map *tagger;

	isl_union_map *dep_flow;
	isl_union_map *dep_false;
	isl_union_map *schedule;

	int n_type;
	struct pet_type **types;
	int n_array;
	struct pet_array **arrays;
	int n_stmt;
	struct pet_stmt **stmts;
};

int ppcg_transform(isl_ctx *ctx, const char *input, FILE *out,
	struct ppcg_options *options,
	__isl_give isl_printer *(*fn)(__isl_take isl_printer *p,
		struct ppcg_scop *scop, void *user), void *user);

#endif
