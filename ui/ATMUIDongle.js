
(async function AutomaticClient() {
	const API = $API;
	this.predict = async (data, props) => {
		return this.api("predict", { payload: { data: data, props: props } });
	};
	// Expects: data && typeof data === "object" && this.props.features
	this.toCSV = (data, default_value=0) => {
		let features = this.props.stats.features;
		return features.join(",") + "\n" + features.map( f => data[f] || ("" + default_value) ).join(",");
	}
	this.api = (command, payload) => {
		payload = payload || {};
		payload.command = command;
		payload.service_uri = this.service_uri;
		payload.api_token = this.api_token;
		return fetch(API, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
			.then(response => {
				return response.json();
			}).catch(e => {
				return { error: "" + e };
			});
	}
	let data = document.getElementById("automatic-ai-app");
	if (data) {
		this.service_uri = data.getAttribute("data-service-uri");
	}
	if (!this.service_uri || this.service_uri.indexOf("/") === -1) {
		this.service_uri = new URLSearchParams(window.location.search).get('service_uri');
	}
	this.api_token = new URLSearchParams(window.location.search).get('api_token') || (process && process.env.AUTOMATIC_API_TOKEN);
	this.props = await this.api("app");
	window.AMC = this;
})();

